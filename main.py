from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets.TimeDataset import SlidingWindowDataset
from evaluate import evaluate_model
from models.FusionModel import FusionAnomalyDetector
from train import train_one_epoch


@dataclass
class TrainConfig:
    window_size: int = 100
    step_size: int = 10
    batch_size: int = 32
    epochs: int = 10
    hidden_dim: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    patience: int = 5
    checkpoint_path: str = "best_model.pth"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def collate_windows(
    batch: list[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack (x, y, start_idx) triples into batched tensors."""
    xs = torch.stack([item[0] for item in batch], dim=0)      # shape: (B, W, C)
    ys = torch.stack([item[1] for item in batch], dim=0)      # shape: (B, W, C)
    starts = torch.stack([item[2] for item in batch], dim=0)  # shape: (B,)
    return xs, ys, starts


def load_array(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    raise ValueError("Only .npy files are supported in this script")


def train_model(
    train_series: torch.Tensor,
    test_series: torch.Tensor,
    test_labels: np.ndarray,
    config: TrainConfig,
) -> Tuple[FusionAnomalyDetector, Dict[str, Any]]:
    device = torch.device(config.device)

    train_dataset = SlidingWindowDataset(
        series=train_series,
        window_size=config.window_size,
        step_size=config.step_size,
    )
    test_dataset = SlidingWindowDataset(
        series=test_series,
        window_size=config.window_size,
        step_size=config.step_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_windows,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_windows,
    )

    model = FusionAnomalyDetector(hidden_dim=config.hidden_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    last_eval: Dict[str, Any] = {}
    best_pa_f1 = float("-inf")
    wait = 0
    for epoch in range(1, config.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=config.grad_clip_norm,
        )

        last_eval = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            total_length=test_series.size(0),
            window_size=config.window_size,
            point_labels=test_labels,
            threshold=None,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_stats['loss']:.6f} | "
            f"NLL: {train_stats['loss_nll']:.6f} | "
            f"NCDE: {train_stats['loss_ncde']:.6f} | "
            f"SACon: {train_stats['loss_sacon']:.6f} | "
            f"Eval PA-F1: {last_eval['point_adjusted_f1']:.6f} | "
            f"{last_eval['vus_pr_name']}: {last_eval['vus_pr']:.6f}"
        )

        current_pa_f1 = float(last_eval["point_adjusted_f1"])
        if current_pa_f1 > best_pa_f1:
            best_pa_f1 = current_pa_f1
            wait = 0
            torch.save(model.state_dict(), config.checkpoint_path)
            print(f"Checkpoint saved to {config.checkpoint_path} (best PA-F1={best_pa_f1:.6f})")
        else:
            wait += 1
            print(f"EarlyStopping counter: {wait}/{config.patience}")
            if wait >= config.patience:
                print(f"Early stopping triggered at epoch {epoch:03d}. Best PA-F1={best_pa_f1:.6f}")
                break

    return model, last_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate FusionAnomalyDetector")
    parser.add_argument("--train", type=str, required=True, help="Path to train series (.npy), shape (T, C)")
    parser.add_argument("--test", type=str, required=True, help="Path to test series (.npy), shape (T, C)")
    parser.add_argument("--test-labels", type=str, required=True, help="Path to test labels (.npy), shape (T,)")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--checkpoint-path", type=str, default="best_model.pth")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_np = load_array(args.train)
    test_np = load_array(args.test)
    labels_np = load_array(args.test_labels)

    if train_np.ndim != 2 or test_np.ndim != 2:
        raise ValueError("train/test arrays must be 2D with shape (T, C)")
    if labels_np.ndim != 1:
        raise ValueError("test labels must be 1D with shape (T,)")
    if labels_np.shape[0] != test_np.shape[0]:
        raise ValueError("test labels length must match test series length")

    config = TrainConfig(
        window_size=args.window_size,
        step_size=args.step_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        patience=args.patience,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
    )

    train_series = torch.from_numpy(train_np).float()
    test_series = torch.from_numpy(test_np).float()
    test_labels = labels_np.astype(np.int64)

    _, final_eval = train_model(
        train_series=train_series,
        test_series=test_series,
        test_labels=test_labels,
        config=config,
    )

    print("Final Eval Summary:")
    print(f"Point-Adjusted F1: {final_eval['point_adjusted_f1']:.6f}")
    print(f"{final_eval['vus_pr_name']}: {final_eval['vus_pr']:.6f}")


if __name__ == "__main__":
    main()

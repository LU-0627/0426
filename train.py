from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fusion_anomaly_detector import FusionAnomalyDetector


class SlidingWindowDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Create (window, start_index) pairs from a multivariate series."""

    def __init__(
        self,
        series: torch.Tensor,
        window_size: int,
        step_size: int,
    ) -> None:
        if series.dim() != 2:
            raise ValueError("series must have shape (Total_Length, Channels)")
        if window_size <= 0 or step_size <= 0:
            raise ValueError("window_size and step_size must be positive")
        if series.size(0) < window_size:
            raise ValueError("Total_Length must be >= window_size")

        self.series = series.float()  # shape: (T, C)
        self.window_size = window_size
        self.step_size = step_size
        self.starts = list(range(0, series.size(0) - window_size + 1, step_size))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[idx]
        end = start + self.window_size
        window = self.series[start:end]  # shape: (W, C)
        start_tensor = torch.tensor(start, dtype=torch.long)  # shape: ()
        return window, start_tensor


def collate_windows(batch: list[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    windows = torch.stack([item[0] for item in batch], dim=0)  # shape: (B, W, C)
    starts = torch.stack([item[1] for item in batch], dim=0)  # shape: (B,)
    return windows, starts


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
) -> Dict[str, float]:
    model.train()
    loss_meter = 0.0
    loss_nll_meter = 0.0
    loss_ncde_meter = 0.0
    loss_sacon_meter = 0.0

    progress = tqdm(dataloader, desc="Train", leave=False)
    for X_batch, _ in progress:
        X_batch = X_batch.to(device)  # shape: (B, W, C)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(X_batch)
        loss = outputs["loss"]  # shape: ()
        loss_nll = outputs["loss_nll"]  # shape: ()
        loss_ncde = outputs["loss_ncde"]  # shape: ()
        loss_sacon = outputs["loss_sacon"]  # shape: ()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        loss_nll_value = float(loss_nll.detach().cpu().item())
        loss_ncde_value = float(loss_ncde.detach().cpu().item())
        loss_sacon_value = float(loss_sacon.detach().cpu().item())

        loss_meter += loss_value
        loss_nll_meter += loss_nll_value
        loss_ncde_meter += loss_ncde_value
        loss_sacon_meter += loss_sacon_value
        progress.set_postfix(
            loss=f"{loss_value:.4f}",
            nll=f"{loss_nll_value:.4f}",
            ncde=f"{loss_ncde_value:.4f}",
            sacon=f"{loss_sacon_value:.4f}",
        )

    denom = max(len(dataloader), 1)
    return {
        "loss": loss_meter / denom,
        "loss_nll": loss_nll_meter / denom,
        "loss_ncde": loss_ncde_meter / denom,
        "loss_sacon": loss_sacon_meter / denom,
    }


def aggregate_window_scores(
    window_scores: torch.Tensor,
    window_starts: torch.Tensor,
    total_length: int,
    window_size: int,
) -> torch.Tensor:
    """Aggregate overlapping window scores to a single timeline score."""
    score_sum = torch.zeros(total_length, dtype=window_scores.dtype)  # shape: (T,)
    score_count = torch.zeros(total_length, dtype=window_scores.dtype)  # shape: (T,)

    num_windows = window_scores.size(0)
    for i in range(num_windows):
        start = int(window_starts[i].item())
        end = start + window_size
        score_sum[start:end] += window_scores[i]  # shape: (W,)
        score_count[start:end] += 1.0  # shape: (W,)

    aggregated = score_sum / score_count.clamp_min(1.0)  # shape: (T,)
    return aggregated


def point_adjust_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Point-adjusted prediction: trigger full anomaly segment if any hit inside segment."""
    y_true = y_true.astype(np.int64)
    adjusted = y_pred.astype(np.int64).copy()

    n = len(y_true)
    idx = 0
    while idx < n:
        if y_true[idx] == 1:
            seg_start = idx
            while idx < n and y_true[idx] == 1:
                idx += 1
            seg_end = idx  # exclusive
            if adjusted[seg_start:seg_end].any():
                adjusted[seg_start:seg_end] = 1
        else:
            idx += 1

    return adjusted


def compute_point_adjusted_f1(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> float:
    y_pred = (y_score >= threshold).astype(np.int64)
    y_pred_adjusted = point_adjust_predictions(y_true=y_true, y_pred=y_pred)
    return float(f1_score(y_true, y_pred_adjusted, zero_division=0))


def compute_vus_pr_or_fallback(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, str]:
    """
    Try VUS-PR; fallback to PR-AUC when VUS library/API is unavailable.
    """
    try:
        from vus.metrics import vus_pr  # type: ignore

        vus_value = float(vus_pr(y_true, y_score))
        return vus_value, "VUS-PR"
    except Exception:
        pr_auc = float(average_precision_score(y_true, y_score))
        return pr_auc, "PR-AUC(Fallback)"


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    total_length: int,
    window_size: int,
    point_labels: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    model.eval()

    all_window_scores = []
    all_window_starts = []

    progress = tqdm(dataloader, desc="Eval", leave=False)
    for X_batch, start_batch in progress:
        X_batch = X_batch.to(device)  # shape: (B, W, C)
        # Causal graph generation uses autograd-based gradients in forward.
        with torch.set_grad_enabled(True):
            outputs = model(X_batch)
        batch_scores = outputs["anomaly_score"].detach().cpu()  # shape: (B, W)

        all_window_scores.append(batch_scores)
        all_window_starts.append(start_batch.cpu())

    window_scores = torch.cat(all_window_scores, dim=0)  # shape: (Nw, W)
    window_starts = torch.cat(all_window_starts, dim=0)  # shape: (Nw,)

    # Required by task: collect all per-time-step scores and concatenate globally.
    global_concat_scores = window_scores.reshape(-1)  # shape: (Nw*W,)

    # Align scores with original timeline for metric computation.
    timeline_scores = aggregate_window_scores(
        window_scores=window_scores,
        window_starts=window_starts,
        total_length=total_length,
        window_size=window_size,
    )  # shape: (T,)

    y_score = timeline_scores.numpy()
    y_true = point_labels.astype(np.int64)

    if threshold is None:
        threshold = float(np.quantile(y_score, 0.95))

    pa_f1 = compute_point_adjusted_f1(y_true=y_true, y_score=y_score, threshold=threshold)
    vus_pr_value, vus_name = compute_vus_pr_or_fallback(y_true=y_true, y_score=y_score)

    return {
        "global_concat_scores": global_concat_scores,
        "timeline_scores": timeline_scores,
        "threshold": threshold,
        "point_adjusted_f1": pa_f1,
        "vus_pr": vus_pr_value,
        "vus_pr_name": vus_name,
    }


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


def load_array(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    raise ValueError("Only .npy files are supported in this script")


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

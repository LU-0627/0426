from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


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
    dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    total_length: int,
    window_size: int,
    point_labels: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Run evaluation and compute anomaly detection metrics."""
    model.eval()

    all_window_scores = []
    all_window_starts = []

    progress = tqdm(dataloader, desc="Eval", leave=False)
    for X_batch, _, start_batch in progress:
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

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Tuple

import torch
import torchcde
from torch import nn

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class DataProcessor(nn.Module):
    """Online model-side preprocess (SFR + interpolation)."""

    def __init__(self) -> None:
        super().__init__()

    def sfr_process(self, X: torch.Tensor) -> torch.Tensor:
        # shape: X (B, W, C)
        mean = X.mean(dim=1, keepdim=True)  # shape: (B, 1, C)
        std = X.std(dim=1, unbiased=False, keepdim=True)  # shape: (B, 1, C)
        X_sfr = (X - mean) / (std + 1e-5)  # shape: (B, W, C)
        X_sfr = torch.clamp(X_sfr, min=-10.0, max=10.0)  # shape: (B, W, C)
        return X_sfr  # shape: (B, W, C)

    def interpolate_process(self, X: torch.Tensor) -> torch.Tensor:
        # shape: X (B, W, C)
        _, W, _ = X.shape
        X_contig = X.contiguous()  # shape: (B, W, C)
        time_grid = torch.linspace(0.0, 1.0, W, device=X.device, dtype=X.dtype)  # shape: (W,)
        coeffs = torchcde.natural_cubic_coeffs(X_contig, t=time_grid)  # shape: (B, W-1, C*4)
        return coeffs  # shape: (B, W-1, C*4)


# ---------------------------------------------------------------------------
# Data-prep utilities below require numpy and pandas at runtime.
# They are NOT imported at module level so that `from util.preprocess import
# DataProcessor` works even when pandas is not installed.
# ---------------------------------------------------------------------------

def _ensure_data_deps():
    """Lazily import numpy and pandas into module globals."""
    global np, pd
    import numpy as np   # noqa: F811
    import pandas as pd  # noqa: F811


def read_table(path: Path) -> "pd.DataFrame":
    _ensure_data_deps()
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, header=None, sep=None, engine="python")


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    return df.ffill().bfill()


def standard_normalize(train: "np.ndarray", test: "np.ndarray") -> Tuple["np.ndarray", "np.ndarray"]:
    _ensure_data_deps()
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.clip(std, a_min=1e-6, a_max=None)
    train_norm = (train - mean) / std
    test_norm = (test - mean) / std
    return train_norm, test_norm


def minmax_normalize(train: "np.ndarray", test: "np.ndarray") -> Tuple["np.ndarray", "np.ndarray"]:
    _ensure_data_deps()
    min_v = train.min(axis=0, keepdims=True)
    max_v = train.max(axis=0, keepdims=True)
    scale = np.clip(max_v - min_v, a_min=1e-6, a_max=None)
    train_norm = (train - min_v) / scale
    test_norm = (test - min_v) / scale
    return train_norm, test_norm


def normalize(train: np.ndarray, test: np.ndarray, method: str) -> Tuple[np.ndarray, np.ndarray]:
    if method == "standard":
        return standard_normalize(train, test)
    if method == "minmax":
        return minmax_normalize(train, test)
    raise ValueError(f"Unsupported normalization method: {method}")


def find_first_existing(base: Path, names: Iterable[str]) -> Optional[Path]:
    for name in names:
        p = base / name
        if p.exists():
            return p
    return None


def read_smd_split(data_dir: Path, split: str, machine_id: Optional[str]) -> np.ndarray:
    direct_txt = data_dir / f"{split}.txt"
    direct_csv = data_dir / f"{split}.csv"
    if direct_txt.exists() or direct_csv.exists():
        df = fill_missing(read_table(direct_txt if direct_txt.exists() else direct_csv))
        return df.to_numpy(dtype=np.float32)

    split_dir = data_dir / split
    if not split_dir.exists() or not split_dir.is_dir():
        raise FileNotFoundError(f"Cannot locate SMD split: {split}")

    if machine_id is not None:
        for ext in (".txt", ".csv"):
            p = split_dir / f"{machine_id}{ext}"
            if p.exists():
                df = fill_missing(read_table(p))
                return df.to_numpy(dtype=np.float32)
        raise FileNotFoundError(f"Cannot find machine file for machine_id={machine_id} in {split_dir}")

    files = sorted(split_dir.glob("*.txt")) + sorted(split_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No files found in {split_dir}")

    arrays = [fill_missing(read_table(p)).to_numpy(dtype=np.float32) for p in files]
    return np.concatenate(arrays, axis=0)


def process_smd_dataset(
    data_dir: str,
    output_dir: str,
    normalization: str = "standard",
    machine_id: Optional[str] = None,
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    _ensure_data_deps()
    root = Path(data_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_arr = read_smd_split(root, split="train", machine_id=machine_id)
    test_arr = read_smd_split(root, split="test", machine_id=machine_id)
    labels_raw = read_smd_split(root, split="test_label", machine_id=machine_id)

    labels_arr = labels_raw[:, 0] if labels_raw.ndim == 2 and labels_raw.shape[1] > 1 else labels_raw.reshape(-1)
    if test_arr.shape[0] != labels_arr.shape[0]:
        raise ValueError(f"SMD test length and label length mismatch: {test_arr.shape[0]} vs {labels_arr.shape[0]}")

    train_arr, test_arr = normalize(train_arr, test_arr, method=normalization)

    labels_arr = (labels_arr > 0).astype(np.int64)
    train_arr = train_arr.astype(np.float32)
    test_arr = test_arr.astype(np.float32)

    np.save(out / "train.npy", train_arr)
    np.save(out / "test.npy", test_arr)
    np.save(out / "test_labels.npy", labels_arr)

    return train_arr, test_arr, labels_arr


def infer_label_series(series: "pd.Series") -> "np.ndarray":
    _ensure_data_deps()
    if pd.api.types.is_numeric_dtype(series):
        return (series.to_numpy() > 0).astype(np.int64)

    text = series.astype(str).str.strip().str.lower()
    attack_like = text.str.contains("attack|anomaly|fault|abnormal", regex=True)
    normal_like = text.str.contains("normal", regex=True)
    labels = np.where(attack_like & ~normal_like, 1, 0)
    if labels.sum() == 0 and (~(text == "normal")).any():
        labels = np.where(text == "normal", 0, 1)
    return labels.astype(np.int64)


def process_swat_dataset(
    data_dir: str,
    output_dir: str,
    normalization: str = "standard",
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    _ensure_data_deps()
    root = Path(data_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_file = find_first_existing(root, ["train.csv", "SWaT_Dataset_Normal_v1.csv"])
    test_file = find_first_existing(root, ["test.csv", "SWaT_Dataset_Attack_v0.csv"])
    label_file = find_first_existing(root, ["test_label.csv", "test_label.txt", "labels.csv", "labels.txt"])
    if train_file is None or test_file is None:
        raise FileNotFoundError("SWaT train/test file not found in data_path")

    train_df = fill_missing(read_table(train_file))
    test_df = fill_missing(read_table(test_file))

    label_candidates = ["Label", "label", "Attack", "attack", "Normal/Attack", "Normal_Attack"]
    label_col_train = next((c for c in label_candidates if c in train_df.columns), None)
    label_col_test = next((c for c in label_candidates if c in test_df.columns), None)

    if label_file is not None:
        labels_series = fill_missing(read_table(label_file)).iloc[:, 0]
    elif label_col_test is not None:
        labels_series = test_df[label_col_test]
    else:
        raise ValueError("Cannot infer SWaT test labels. Provide a label file or label column in test CSV.")

    if label_col_train is not None:
        train_df = train_df.drop(columns=[label_col_train])
    if label_col_test is not None:
        test_df = test_df.drop(columns=[label_col_test])

    train_num = train_df.select_dtypes(include=[np.number]).copy()
    test_num = test_df.select_dtypes(include=[np.number]).copy()
    common_cols = [c for c in train_num.columns if c in test_num.columns]
    if not common_cols:
        raise ValueError("No common numeric feature columns between SWaT train and test.")

    train_arr = train_num[common_cols].to_numpy(dtype=np.float32)
    test_arr = test_num[common_cols].to_numpy(dtype=np.float32)
    labels_arr = infer_label_series(labels_series)
    if test_arr.shape[0] != labels_arr.shape[0]:
        raise ValueError(f"SWaT test length and label length mismatch: {test_arr.shape[0]} vs {labels_arr.shape[0]}")

    train_arr, test_arr = normalize(train_arr, test_arr, method=normalization)

    train_arr = train_arr.astype(np.float32)
    test_arr = test_arr.astype(np.float32)
    labels_arr = labels_arr.astype(np.int64)

    np.save(out / "train.npy", train_arr)
    np.save(out / "test.npy", test_arr)
    np.save(out / "test_labels.npy", labels_arr)

    return train_arr, test_arr, labels_arr

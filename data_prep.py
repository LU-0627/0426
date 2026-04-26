from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    """Read csv/txt robustly with automatic delimiter inference."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, header=None, sep=None, engine="python")


def _ffill_bfill(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values by forward fill then backward fill."""
    return df.ffill().bfill()


def _standard_normalize(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.clip(std, a_min=1e-6, a_max=None)
    train_norm = (train - mean) / std
    test_norm = (test - mean) / std
    return train_norm, test_norm


def _minmax_normalize(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    min_v = train.min(axis=0, keepdims=True)
    max_v = train.max(axis=0, keepdims=True)
    scale = np.clip(max_v - min_v, a_min=1e-6, a_max=None)
    train_norm = (train - min_v) / scale
    test_norm = (test - min_v) / scale
    return train_norm, test_norm


def _normalize(
    train: np.ndarray,
    test: np.ndarray,
    method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if method == "standard":
        return _standard_normalize(train, test)
    if method == "minmax":
        return _minmax_normalize(train, test)
    raise ValueError(f"Unsupported normalization method: {method}")


def _find_first_existing(base: Path, names: Iterable[str]) -> Optional[Path]:
    for name in names:
        p = base / name
        if p.exists():
            return p
    return None


def _read_smd_split(
    data_dir: Path,
    split: str,
    machine_id: Optional[str],
) -> np.ndarray:
    """
    Read SMD split for train/test/test_label.

    Supported layouts:
    1) data_dir/split/<machine_id>.txt (official SMD style)
    2) data_dir/split/*.txt (concat all machine files if machine_id not provided)
    3) data_dir/<split>.txt or data_dir/<split>.csv
    """
    direct_txt = data_dir / f"{split}.txt"
    direct_csv = data_dir / f"{split}.csv"
    if direct_txt.exists() or direct_csv.exists():
        df = _read_table(direct_txt if direct_txt.exists() else direct_csv)
        df = _ffill_bfill(df)
        return df.to_numpy(dtype=np.float32)

    split_dir = data_dir / split
    if not split_dir.exists() or not split_dir.is_dir():
        raise FileNotFoundError(f"Cannot locate SMD split: {split}")

    if machine_id is not None:
        for ext in (".txt", ".csv"):
            p = split_dir / f"{machine_id}{ext}"
            if p.exists():
                df = _read_table(p)
                df = _ffill_bfill(df)
                return df.to_numpy(dtype=np.float32)
        raise FileNotFoundError(f"Cannot find machine file for machine_id={machine_id} in {split_dir}")

    files = sorted(split_dir.glob("*.txt")) + sorted(split_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No files found in {split_dir}")

    arrays = []
    for p in files:
        df = _read_table(p)
        df = _ffill_bfill(df)
        arrays.append(df.to_numpy(dtype=np.float32))
    return np.concatenate(arrays, axis=0)


def process_smd_dataset(
    data_dir: str,
    output_dir: str,
    normalization: str = "standard",
    machine_id: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process SMD dataset and export train/test/labels npy files.

    Returns:
        train_arr: shape (T_train, C)
        test_arr: shape (T_test, C)
        labels_arr: shape (T_test,)
    """
    root = Path(data_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_arr = _read_smd_split(root, split="train", machine_id=machine_id)
    test_arr = _read_smd_split(root, split="test", machine_id=machine_id)

    labels_raw = _read_smd_split(root, split="test_label", machine_id=machine_id)
    if labels_raw.ndim == 2 and labels_raw.shape[1] > 1:
        labels_arr = labels_raw[:, 0]
    else:
        labels_arr = labels_raw.reshape(-1)

    if test_arr.shape[0] != labels_arr.shape[0]:
        raise ValueError(
            f"SMD test length and label length mismatch: {test_arr.shape[0]} vs {labels_arr.shape[0]}"
        )

    train_arr, test_arr = _normalize(train_arr, test_arr, method=normalization)

    labels_arr = (labels_arr > 0).astype(np.int64)
    train_arr = train_arr.astype(np.float32)
    test_arr = test_arr.astype(np.float32)

    np.save(out / "train.npy", train_arr)
    np.save(out / "test.npy", test_arr)
    np.save(out / "test_labels.npy", labels_arr)

    return train_arr, test_arr, labels_arr


def _infer_label_series(series: pd.Series) -> np.ndarray:
    """Convert label column into binary anomaly labels (0/1)."""
    if pd.api.types.is_numeric_dtype(series):
        return (series.to_numpy() > 0).astype(np.int64)

    text = series.astype(str).str.strip().str.lower()
    attack_like = text.str.contains("attack|anomaly|fault|abnormal", regex=True)
    normal_like = text.str.contains("normal", regex=True)
    labels = np.where(attack_like & ~normal_like, 1, 0)

    # If labels are values like 'Attack' / 'Normal', above works.
    # Fallback: anything not exactly 'normal' is treated as anomaly.
    if labels.sum() == 0 and (~(text == "normal")).any():
        labels = np.where(text == "normal", 0, 1)
    return labels.astype(np.int64)


def process_swat_dataset(
    data_dir: str,
    output_dir: str,
    normalization: str = "standard",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process SWaT dataset and export train/test/labels npy files.

    Expected files in data_dir (flexible names):
    - train CSV: train.csv or SWaT_Dataset_Normal_v1.csv
    - test CSV: test.csv or SWaT_Dataset_Attack_v0.csv
    - optional label CSV/TXT: test_label.csv / test_label.txt / labels.csv
    """
    root = Path(data_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_file = _find_first_existing(root, ["train.csv", "SWaT_Dataset_Normal_v1.csv"])
    test_file = _find_first_existing(root, ["test.csv", "SWaT_Dataset_Attack_v0.csv"])
    label_file = _find_first_existing(root, ["test_label.csv", "test_label.txt", "labels.csv", "labels.txt"])

    if train_file is None or test_file is None:
        raise FileNotFoundError("SWaT train/test file not found in data_path")

    train_df = _ffill_bfill(_read_table(train_file))
    test_df = _ffill_bfill(_read_table(test_file))

    label_candidates = ["Label", "label", "Attack", "attack", "Normal/Attack", "Normal_Attack"]
    label_col_train = next((c for c in label_candidates if c in train_df.columns), None)
    label_col_test = next((c for c in label_candidates if c in test_df.columns), None)

    if label_file is not None:
        labels_df = _ffill_bfill(_read_table(label_file))
        labels_series = labels_df.iloc[:, 0]
    elif label_col_test is not None:
        labels_series = test_df[label_col_test]
    else:
        raise ValueError("Cannot infer SWaT test labels. Provide a label file or label column in test CSV.")

    if label_col_train is not None:
        train_df = train_df.drop(columns=[label_col_train])
    if label_col_test is not None:
        test_df = test_df.drop(columns=[label_col_test])

    # Keep numeric features only and align test columns to train columns.
    train_num = train_df.select_dtypes(include=[np.number]).copy()
    test_num = test_df.select_dtypes(include=[np.number]).copy()

    common_cols = [c for c in train_num.columns if c in test_num.columns]
    if not common_cols:
        raise ValueError("No common numeric feature columns between SWaT train and test.")

    train_arr = train_num[common_cols].to_numpy(dtype=np.float32)
    test_arr = test_num[common_cols].to_numpy(dtype=np.float32)
    labels_arr = _infer_label_series(labels_series)

    if test_arr.shape[0] != labels_arr.shape[0]:
        raise ValueError(
            f"SWaT test length and label length mismatch: {test_arr.shape[0]} vs {labels_arr.shape[0]}"
        )

    train_arr, test_arr = _normalize(train_arr, test_arr, method=normalization)

    np.save(out / "train.npy", train_arr.astype(np.float32))
    np.save(out / "test.npy", test_arr.astype(np.float32))
    np.save(out / "test_labels.npy", labels_arr.astype(np.int64))

    return train_arr.astype(np.float32), test_arr.astype(np.float32), labels_arr.astype(np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess industrial MTSAD datasets to .npy files")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["smd", "swat"])
    parser.add_argument("--data_path", type=str, required=True, help="Dataset root path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save train/test/test_labels npy")
    parser.add_argument("--normalization", type=str, default="standard", choices=["standard", "minmax"])
    parser.add_argument(
        "--machine_id",
        type=str,
        default=None,
        help="Optional machine id for SMD, e.g. machine-1-1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset_name == "smd":
        train_arr, test_arr, labels_arr = process_smd_dataset(
            data_dir=args.data_path,
            output_dir=args.output_dir,
            normalization=args.normalization,
            machine_id=args.machine_id,
        )
    else:
        train_arr, test_arr, labels_arr = process_swat_dataset(
            data_dir=args.data_path,
            output_dir=args.output_dir,
            normalization=args.normalization,
        )

    print("Data preparation finished.")
    print(f"train.npy shape: {train_arr.shape}")
    print(f"test.npy shape: {test_arr.shape}")
    print(f"test_labels.npy shape: {labels_arr.shape}")
    print(f"test_labels dtype: {labels_arr.dtype}")


if __name__ == "__main__":
    main()

"""Quick sanity check after the 4-point refactoring."""
from __future__ import annotations

import sys

import torch


def main() -> None:
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    errors = []

    # 1. Check Dataset returns 3-tuple with start_idx
    print("\n[1] datasets.TimeDataset.SlidingWindowDataset ...")
    from datasets.TimeDataset import SlidingWindowDataset

    series = torch.randn(200, 5)
    ds = SlidingWindowDataset(series, window_size=50, step_size=10)
    sample = ds[0]
    assert len(sample) == 3, f"Expected 3-tuple, got {len(sample)}"
    x, y, start_idx = sample
    assert x.shape == (50, 5), f"x shape: {x.shape}"
    assert y.shape == (50, 5), f"y shape: {y.shape}"
    assert start_idx.dtype == torch.long, f"start_idx dtype: {start_idx.dtype}"
    assert start_idx.item() == 0, f"start_idx value: {start_idx.item()}"
    # Check a later sample
    x2, y2, si2 = ds[3]
    assert si2.item() == 30, f"start_idx[3] should be 30, got {si2.item()}"
    print("  [PASS] Returns (x, y, start_idx) correctly")

    # 2. Check train.py has train_one_epoch and nothing else
    print("\n[2] train.py ...")
    import train
    assert hasattr(train, "train_one_epoch"), "Missing train_one_epoch"
    for forbidden in ["SlidingWindowDataset", "collate_windows", "aggregate_window_scores",
                       "evaluate_model", "TrainConfig", "parse_args", "main"]:
        if hasattr(train, forbidden):
            errors.append(f"train.py still has '{forbidden}'")
            print(f"  [FAIL] train.py still exports '{forbidden}'")
    if not errors:
        print("  [PASS] train.py is clean (only train_one_epoch)")

    # 3. Check evaluate.py has all eval functions
    print("\n[3] evaluate.py ...")
    import evaluate
    for fn in ["aggregate_window_scores", "point_adjust_predictions",
                "compute_point_adjusted_f1", "compute_vus_pr_or_fallback", "evaluate_model"]:
        if not hasattr(evaluate, fn):
            errors.append(f"evaluate.py missing '{fn}'")
            print(f"  [FAIL] evaluate.py missing '{fn}'")
    if not any("evaluate.py" in e for e in errors):
        print("  [PASS] evaluate.py has all evaluation functions")

    # 4. Check main.py imports
    print("\n[4] main.py imports ...")
    import main
    for attr in ["TrainConfig", "collate_windows", "load_array", "train_model", "parse_args", "main"]:
        if not hasattr(main, attr):
            errors.append(f"main.py missing '{attr}'")
            print(f"  [FAIL] main.py missing '{attr}'")
    if not any("main.py" in e for e in errors):
        print("  [PASS] main.py has all orchestration functions")

    # 5. Quick forward pass through refactored model
    print("\n[5] Forward + backward (refactored model) ...")
    from models.FusionModel import FusionAnomalyDetector
    B, W, C, H = 2, 50, 5, 32
    torch.manual_seed(42)
    model = FusionAnomalyDetector(hidden_dim=H)
    X = torch.randn(B, W, C)
    out = model(X)
    out["loss"].backward()
    print(f"  [PASS] loss={out['loss'].item():.6f}, backward OK")

    # 6. DataLoader round-trip with collate_windows
    print("\n[6] DataLoader with collate_windows ...")
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=main.collate_windows)
    batch = next(iter(loader))
    assert len(batch) == 3, f"Batch should be 3-tuple, got {len(batch)}"
    bx, by, bstarts = batch
    assert bx.shape == (4, 50, 5)
    assert by.shape == (4, 50, 5)
    assert bstarts.shape == (4,)
    print(f"  [PASS] batch shapes: x={tuple(bx.shape)}, y={tuple(by.shape)}, starts={tuple(bstarts.shape)}")

    # Summary
    print("\n" + "=" * 50)
    if errors:
        print(f"  FAILED ({len(errors)} error(s)):")
        for e in errors:
            print(f"    - {e}")
        sys.exit(1)
    else:
        print("  ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

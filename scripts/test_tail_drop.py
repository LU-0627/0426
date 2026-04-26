import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from datasets.TimeDataset import SlidingWindowDataset

# Case 1: tail NOT aligned (105 - 50 = 55, 55 % 20 != 0)
ds = SlidingWindowDataset(torch.randn(105, 3), window_size=50, step_size=20)
print(f"starts: {ds.starts}")
assert ds.starts[-1] == 55, f"Expected last start=55, got {ds.starts[-1]}"
assert ds.starts[-1] + 50 == 105, "Tail not fully covered!"

# Case 2: tail already aligned (100 - 50 = 50, 50 % 10 == 0)
ds2 = SlidingWindowDataset(torch.randn(100, 3), window_size=50, step_size=10)
assert ds2.starts[-1] == 50
assert ds2.starts.count(50) == 1, "Duplicate tail entry!"

# Case 3: exact fit
ds3 = SlidingWindowDataset(torch.randn(50, 3), window_size=50, step_size=10)
assert ds3.starts == [0], f"Expected [0], got {ds3.starts}"

print("[PASS] tail-drop fix verified")

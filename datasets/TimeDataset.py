from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Create (x, y) window pairs from a multivariate series."""

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

        self.series = series.float()
        self.window_size = window_size
        self.step_size = step_size
        self.starts = list(range(0, series.size(0) - window_size + 1, step_size))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[idx]
        end = start + self.window_size
        x = self.series[start:end]
        y = x.clone()
        return x, y

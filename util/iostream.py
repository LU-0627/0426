from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def log_info(message: str) -> None:
    print(message)


def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    ckpt = Path(path)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)


def load_checkpoint(model: torch.nn.Module, path: str, map_location: torch.device) -> None:
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)


def summarize_metrics(metrics: Dict[str, Any]) -> str:
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.6f}")
        else:
            parts.append(f"{k}={v}")
    return " | ".join(parts)

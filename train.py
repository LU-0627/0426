from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
) -> Dict[str, float]:
    """Run a single training epoch and return average loss components."""
    model.train()
    loss_meter = 0.0
    loss_nll_meter = 0.0
    loss_ncde_meter = 0.0
    loss_sacon_meter = 0.0

    progress = tqdm(dataloader, desc="Train", leave=False)
    for X_batch, _, _ in progress:
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

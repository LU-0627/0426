from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from models.layers import CausalGraphGenerator, NCDEBranch, TimeSpatialTransformer
from util.preprocess import DataProcessor


class JointLoss(nn.Module):
    """Joint objective: MTS-NLL + NCDE derivative consistency + SACon regularization."""

    def __init__(self, lambda_dyn: float = 1.0, lambda_sacon: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.lambda_dyn = lambda_dyn
        self.lambda_sacon = lambda_sacon
        self.eps = eps
        self._debug_printed = False

    def forward(
        self,
        mu: torch.Tensor,  # shape: (B, W, C)
        sigma_sq: torch.Tensor,  # shape: (B, W, C)
        X_sfr: torch.Tensor,  # shape: (B, W, C)
        sd: torch.Tensor,  # shape: (B, W, H)
        sd_hat: torch.Tensor,  # shape: (B, W, H)
        sacon: torch.Tensor,  # shape: (B, W, C)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma_safe = sigma_sq.clamp_min(self.eps)  # shape: (B, W, C)

        # MTS-NLL = 0.5*log(sigma^2) + (x-mu)^2 / (2*sigma^2)
        nll_var_term = 0.5 * torch.log(sigma_safe)  # shape: (B, W, C)
        rec_numerator = (X_sfr - mu).pow(2).clamp(max=1e4)  # shape: (B, W, C)
        nll_rec_term = 0.5 * rec_numerator / sigma_safe  # shape: (B, W, C)
        mts_nll_loss = (nll_var_term + nll_rec_term).mean()  # shape: ()

        # NCDE derivative consistency.
        ncde_derivative_loss = F.smooth_l1_loss(sd_hat, sd, beta=1.0)  # shape: ()

        # SACon regularization: maximize contribution -> negative mean.
        sacon_reg = -sacon.mean()  # shape: ()

        if (
            (torch.isnan(mts_nll_loss) or torch.isinf(mts_nll_loss)
             or torch.isnan(ncde_derivative_loss) or torch.isinf(ncde_derivative_loss))
            and not self._debug_printed
        ):
            print(
                f"[DEBUG] Raw NLL: {mts_nll_loss.item()}, "
                f"Raw NCDE: {ncde_derivative_loss.item()}, "
                f"Raw SACon: {sacon_reg.item()}"
            )
            self._debug_printed = True

        mts_nll_loss = torch.nan_to_num(mts_nll_loss, nan=0.0, posinf=1e6, neginf=0.0)  # shape: ()
        ncde_derivative_loss = torch.nan_to_num(ncde_derivative_loss, nan=0.0, posinf=1e6, neginf=0.0)  # shape: ()
        sacon_reg = torch.nan_to_num(sacon_reg, nan=0.0, posinf=1e6, neginf=-1e6)  # shape: ()

        loss = mts_nll_loss + self.lambda_dyn * ncde_derivative_loss + self.lambda_sacon * sacon_reg  # shape: ()
        return loss, mts_nll_loss, ncde_derivative_loss, sacon_reg  # shape: (), (), (), ()


class FusionAnomalyDetector(nn.Module):
    """Fusion MTSAD model with SFR, causality graph, NCDE, and SACon."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lambda1_uncertainty = 1.0
        self.lambda2_derivative = 0.5

        self.data_processor = DataProcessor()
        self.causal_graph_generator = CausalGraphGenerator()
        self.time_spatial_transformer = TimeSpatialTransformer()
        self.ncde_branch: NCDEBranch | None = None
        self.sd_predictor: nn.Linear | None = None
        self.joint_loss = JointLoss(lambda_dyn=0.001)

    def _lazy_init_ncde(self, input_dim: int, device: torch.device, dtype: torch.dtype) -> None:
        if self.ncde_branch is None:
            self.ncde_branch = NCDEBranch(input_dim=input_dim, hidden_dim=self.hidden_dim)
            self.ncde_branch.to(device=device, dtype=dtype)
        if self.sd_predictor is None:
            self.sd_predictor = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.sd_predictor.weight.data *= 0.01
            if self.sd_predictor.bias is not None:
                self.sd_predictor.bias.data *= 0.01
            self.sd_predictor.to(device=device, dtype=dtype)

    def forward(
        self,
        X: torch.Tensor,  # shape: (B, W, C)
    ) -> Dict[str, torch.Tensor]:
        self._lazy_init_ncde(input_dim=X.shape[-1], device=X.device, dtype=X.dtype)

        # Stream A: SFR -> causality graph -> time-spatial transformer.
        X_sfr = self.data_processor.sfr_process(X)  # shape: (B, W, C)
        adj_matrix = self.causal_graph_generator(X_sfr)  # shape: (B, C, C)
        mu, sigma_sq, sacon = self.time_spatial_transformer(X_sfr, adj_matrix)  # shape: (B, W, C), (B, W, C), (B, W, C)

        # Stream B: interpolation coefficients -> NCDE state derivative.
        coeffs = self.data_processor.interpolate_process(X)  # shape: (B, W-1, C*4)
        sd = self.ncde_branch(coeffs)  # shape: (B, W, H)

        # NCDE derivative predictor for dynamic consistency.
        sd_hat = self.sd_predictor(sd)  # shape: (B, W, H)
        sd_hat = torch.clamp(sd_hat, min=-1e3, max=1e3)  # shape: (B, W, H)

        loss, loss_nll, loss_ncde, loss_sacon = self.joint_loss(mu, sigma_sq, X_sfr, sd, sd_hat, sacon)  # shape: (), (), (), ()

        # Fused anomaly score from uncertainty deviation and derivative deviation.
        sigma_safe = sigma_sq.clamp_min(1e-6)  # shape: (B, W, C)
        score_uncertainty = 0.5 * (X_sfr - mu).pow(2) / sigma_safe  # shape: (B, W, C)
        score_uncertainty_t = score_uncertainty.sum(dim=2)  # shape: (B, W)

        score_derivative_t = (sd - sd_hat).pow(2).sum(dim=2)  # shape: (B, W)

        anomaly_score = (
            self.lambda1_uncertainty * score_uncertainty_t
            + self.lambda2_derivative * score_derivative_t
        )  # shape: (B, W)
        anomaly_score = torch.nan_to_num(anomaly_score, nan=0.0, posinf=1e6, neginf=0.0)  # shape: (B, W)

        return {
            "mu": mu,
            "sigma_sq": sigma_sq,
            "adj_matrix": adj_matrix,
            "sacon": sacon,
            "coeffs": coeffs,
            "sd": sd,
            "sd_hat": sd_hat,
            "loss": loss,
            "loss_nll": loss_nll,
            "loss_ncde": loss_ncde,
            "loss_sacon": loss_sacon,
            "anomaly_score": anomaly_score,
        }  # shape: {mu:(B,W,C), sigma_sq:(B,W,C), adj_matrix:(B,C,C), sacon:(B,W,C), coeffs:(B,W-1,C*4), sd:(B,W,H), sd_hat:(B,W,H), loss:(), loss_nll:(), loss_ncde:(), loss_sacon:(), anomaly_score:(B,W)}

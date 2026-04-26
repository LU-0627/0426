from __future__ import annotations

from typing import Dict, Tuple

import torch
import torchcde
from torch import nn
from torch.nn import functional as F


class DataProcessor(nn.Module):
    """Data preprocessing module for dual-stream inputs."""

    def __init__(self) -> None:
        super().__init__()

    def sfr_process(self, X: torch.Tensor) -> torch.Tensor:
        """
        Statistical feature removal (SFR): removes local mean and variance.

        Args:
            X: Raw input series. Shape: (B, W, C)

        Returns:
            Standardized tensor. Shape: (B, W, C)
        """
        # Calculate mean and std along the time window dimension (dim=1)
        mean = X.mean(dim=1, keepdim=True)  # shape: (B, 1, C)
        std = X.std(dim=1, unbiased=False, keepdim=True)  # shape: (B, 1, C)

        # Remove statistical features to force model to learn dependencies
        X_sfr = (X - mean) / (std + 1e-5)  # shape: (B, W, C)
        X_sfr = torch.clamp(X_sfr, min=-10.0, max=10.0)  # shape: (B, W, C)
        return X_sfr  # shape: (B, W, C)

    def interpolate_process(self, X: torch.Tensor) -> torch.Tensor:
        """
        Interpolation placeholder for continuous-path features.

        Args:
            X: Raw input series. Shape: (B, W, C)

        Returns:
            Natural cubic spline coefficients. Shape: (B, W-1, C*4)
        """
        _, W, _ = X.shape  # shape: (B, W, C)
        X_contig = X.contiguous()  # shape: (B, W, C)
        time_grid = torch.linspace(0.0, 1.0, W, device=X.device, dtype=X.dtype)  # shape: (W,)
        coeffs = torchcde.natural_cubic_coeffs(X_contig, t=time_grid)  # shape: (B, W-1, C*4)
        return coeffs  # shape: (B, W-1, C*4)


class CDEFunc(nn.Module):
    """Vector field f(t, z) for NCDE dynamics."""

    def __init__(self, input_dim: int, hidden_dim: int, mlp_hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, hidden_dim * input_dim),
        )
        # Shrink the final layer initialization to stabilize early NCDE dynamics.
        self.net[-1].weight.data *= 0.1
        if self.net[-1].bias is not None:
            self.net[-1].bias.data *= 0.1

    def forward(
        self,
        t: torch.Tensor,  # shape: ()
        z: torch.Tensor,  # shape: (B, H)
    ) -> torch.Tensor:
        _ = t  # shape: ()
        B, _ = z.shape  # shape: (B, H)
        field_flat = self.net(z)  # shape: (B, H*C)
        field_flat = torch.tanh(field_flat)  # shape: (B, H*C)
        field = field_flat.view(B, self.hidden_dim, self.input_dim)  # shape: (B, H, C)
        return field  # shape: (B, H, C)


class CausalGraphGenerator(nn.Module):
    """Dynamic Granger-causality graph generator."""

    def __init__(self, hidden_dim: int = 128, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2

        # Non-linear predictor for per-channel temporal reconstruction.
        self.predictor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=1, kernel_size=1),
        )

        # Learnable hard-threshold parameter for graph sparsification.
        self.h = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(
        self,
        X: torch.Tensor,  # shape: (B, W, C)
    ) -> torch.Tensor:
        B, W, C = X.shape  # shape: (B, W, C)

        # Ensure X participates in autograd-based causal gradient computation.
        if X.requires_grad:
            X_req = X  # shape: (B, W, C)
        else:
            X_req = X.detach().clone().requires_grad_(True)  # shape: (B, W, C)

        # Channel-wise temporal prediction with shared 1D-CNN predictor.
        X_ch = X_req.permute(0, 2, 1).contiguous()  # shape: (B, C, W)
        X_ch = X_ch.view(B * C, 1, W)  # shape: (B*C, 1, W)
        pred_ch = self.predictor(X_ch)  # shape: (B*C, 1, W)
        pred = pred_ch.view(B, C, W).permute(0, 2, 1).contiguous()  # shape: (B, W, C)

        # Gradient-based channel influence integration: A[src, tgt].
        influence_list = []
        for tgt_idx in range(C):
            target_scalar = pred[:, :, tgt_idx].sum()  # shape: ()
            grad_tgt = torch.autograd.grad(
                outputs=target_scalar,
                inputs=X_req,
                retain_graph=True,
                create_graph=self.training,
                allow_unused=False,
            )[0]  # shape: (B, W, C)
            influence_src = grad_tgt.abs().mean(dim=1)  # shape: (B, C)
            influence_list.append(influence_src)

        A = torch.stack(influence_list, dim=2)  # shape: (B, C, C)

        # Asymmetric differencing to suppress bidirectional similarity noise.
        A_diff = A - A.transpose(1, 2)  # shape: (B, C, C)

        # ReLU + learnable hard-threshold for sparse directed graph.
        h_val = self.h.to(device=A_diff.device, dtype=A_diff.dtype)  # shape: ()
        adj_matrix = torch.relu(A_diff - h_val)  # shape: (B, C, C)

        # Remove self-loops on the diagonal.
        eye = torch.eye(C, device=adj_matrix.device, dtype=torch.bool).unsqueeze(0)  # shape: (1, C, C)
        adj_matrix = adj_matrix.masked_fill(eye, 0.0)  # shape: (B, C, C)
        return adj_matrix  # shape: (B, C, C)


class TimeSpatialTransformer(nn.Module):
    """Time-spatial transformer branch with sub-neighborhood attention placeholder."""

    def __init__(self, k1: int = 20, k2: int = 30) -> None:
        super().__init__()
        self.k1 = k1
        self.k2 = k2

        # Learnable temperature for temporal attention sharpening/smoothing.
        self.log_temp = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # Lightweight linear mappings (channel-agnostic by scalar projection).
        self.q_proj = nn.Linear(1, 1)
        self.k_proj = nn.Linear(1, 1)
        self.v_proj = nn.Linear(1, 1)
        self.mu_head = nn.Linear(1, 1)
        self.sigma_head = nn.Linear(1, 1)

    def forward(
        self,
        X_sfr: torch.Tensor,  # shape: (B, W, C)
        adj_matrix: torch.Tensor,  # shape: (B, C, C)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, W, C = X_sfr.shape  # shape: (B, W, C)

        # Spatial graph message passing along directed causal edges.
        adj_nonneg = torch.relu(adj_matrix)  # shape: (B, C, C)
        in_deg = adj_nonneg.sum(dim=1, keepdim=True).clamp(min=1e-4)  # shape: (B, 1, C)
        adj_norm = adj_nonneg / in_deg  # shape: (B, C, C)
        X_spatial = torch.einsum("bwi,bij->bwj", X_sfr, adj_norm)  # shape: (B, W, C)

        # Build sub-adjacent temporal mask with windows [t-K2, t-K1] and [t+K1, t+K2].
        time_idx = torch.arange(W, device=X_sfr.device)  # shape: (W,)
        delta = (time_idx[None, :] - time_idx[:, None]).abs()  # shape: (W, W)
        sub_adj_mask = (delta >= self.k1) & (delta <= self.k2)  # shape: (W, W)

        # If a target time has no valid sub-neighborhood, fallback to self-attend.
        has_valid = sub_adj_mask.any(dim=1)  # shape: (W,)
        if not bool(has_valid.all()):
            fallback_eye = torch.eye(W, device=X_sfr.device, dtype=torch.bool)  # shape: (W, W)
            sub_adj_mask = torch.where(has_valid[:, None], sub_adj_mask, fallback_eye)  # shape: (W, W)

        # Temporal linear attention with learnable temperature.
        temporal_token = X_spatial.mean(dim=2, keepdim=True)  # shape: (B, W, 1)
        q = self.q_proj(temporal_token)  # shape: (B, W, 1)
        k = self.k_proj(temporal_token)  # shape: (B, W, 1)
        v = self.v_proj(X_spatial.unsqueeze(-1)).squeeze(-1)  # shape: (B, W, C)

        temp = torch.nn.functional.softplus(self.log_temp) + 1e-4  # shape: ()
        attn_logits = torch.matmul(q, k.transpose(1, 2)) / temp  # shape: (B, W, W)

        attn_mask = sub_adj_mask.unsqueeze(0).expand(B, -1, -1)  # shape: (B, W, W)
        attn_logits = attn_logits.masked_fill(~attn_mask, -1e9)  # shape: (B, W, W)
        attn_weights = torch.softmax(attn_logits, dim=-1)  # shape: (B, W, W)
        attn_weights = attn_weights * attn_mask.to(dtype=attn_weights.dtype)  # shape: (B, W, W)
        attn_den = attn_weights.sum(dim=-1, keepdim=True) + 1e-6  # shape: (B, W, 1)
        attn_weights = attn_weights / attn_den  # shape: (B, W, W)

        # Sub-neighborhood attention contribution (SACon) for regularization.
        sacon_scalar = attn_weights.sum(dim=-1, keepdim=True)  # shape: (B, W, 1)
        sacon = sacon_scalar.expand(-1, -1, C)  # shape: (B, W, C)

        # Temporal aggregation and residual fusion.
        X_temporal = torch.matmul(attn_weights, v)  # shape: (B, W, C)
        X_fused = X_temporal + X_spatial  # shape: (B, W, C)

        # Linear heads for Gaussian reconstruction parameters.
        mu = self.mu_head(X_fused.unsqueeze(-1)).squeeze(-1)  # shape: (B, W, C)
        mu = torch.clamp(mu, min=-20.0, max=20.0)  # shape: (B, W, C)
        sigma_raw = self.sigma_head(X_fused.unsqueeze(-1)).squeeze(-1)  # shape: (B, W, C)
        sigma_sq = torch.nn.functional.softplus(sigma_raw) + 0.1  # shape: (B, W, C)
        return mu, sigma_sq, sacon  # shape: (B, W, C), (B, W, C), (B, W, C)


class NCDEBranch(nn.Module):
    """NCDE branch placeholder."""

    def __init__(self, input_dim: int, hidden_dim: int, mlp_hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.func = CDEFunc(input_dim=input_dim, hidden_dim=hidden_dim, mlp_hidden_dim=mlp_hidden_dim)
        self.z0_proj = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        coeffs: torch.Tensor,  # shape: (B, W-1, C*4)
    ) -> torch.Tensor:
        X_path = torchcde.CubicSpline(coeffs)  # shape: continuous path X(t)
        times = X_path.grid_points  # shape: (W,)
        W = times.shape[0]  # shape: (W,)

        x0 = X_path.evaluate(times[0])  # shape: (B, C)
        z0 = self.z0_proj(x0)  # shape: (B, H)

        z_traj = torchcde.cdeint(
            X=X_path,
            z0=z0,
            func=self.func,
            t=times,
            method="rk4",
            adjoint=False,
        )  # shape: (W, B, H)

        if z_traj.shape[0] == W:
            z_wbh = z_traj  # shape: (W, B, H)
        else:
            z_wbh = z_traj.transpose(0, 1).contiguous()  # shape: (W, B, H)

        x_deriv = X_path.derivative(times)  # shape: (W, B, C)
        if x_deriv.shape[0] == W:
            xdot_wbc = x_deriv  # shape: (W, B, C)
        else:
            xdot_wbc = x_deriv.transpose(0, 1).contiguous()  # shape: (W, B, C)

        field_seq = []
        for w_idx in range(W):
            f_w = self.func(times[w_idx], z_wbh[w_idx])  # shape: (B, H, C)
            field_seq.append(f_w)
        field_whc = torch.stack(field_seq, dim=0)  # shape: (W, B, H, C)

        sd_wbh = torch.einsum("wbhc,wbc->wbh", field_whc, xdot_wbc)  # shape: (W, B, H)
        sd = sd_wbh.permute(1, 0, 2).contiguous()  # shape: (B, W, H)
        sd = self.layer_norm(sd)  # shape: (B, W, H)
        sd = torch.clamp(sd, min=-1e3, max=1e3)  # shape: (B, W, H)
        return sd  # shape: (B, W, H)


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
    """Fusion MTSAD skeleton integrating SFR, causality, NCDE, and joint optimization."""

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


if __name__ == "__main__":
    B, W, C, H = 4, 16, 8, 32
    X_demo = torch.randn(B, W, C)

    model = FusionAnomalyDetector(hidden_dim=H)
    outputs = model(X_demo)

    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {tuple(value.shape)}")

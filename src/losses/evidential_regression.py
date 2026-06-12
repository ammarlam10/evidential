"""
Evidential regression loss for Normal Inverse-Gamma (NIG) output heads.

Implements the two-term loss from Amini et al., "Deep Evidential Regression",
NeurIPS 2020 (https://www.mit.edu/~amini/pubs/pdf/deep-evidential-regression.pdf).

Given a model that outputs NIG hyperparameters (γ, ν, α, β) per pixel per target,
the total loss per pixel is:

    L = L_NLL + λ · L_R

where:

    Ω     = 2β(1 + ν)
    L_NLL = 0.5·log(π/ν) − α·log(Ω)
            + (α + 0.5)·log((y − γ)²·ν + Ω)
            + lgamma(α) − lgamma(α + 0.5)          [Eq. 8 / S1.2]

    L_R   = |y − γ| · (2ν + α)                     [Eq. 9]

Expected input channel layout from UNetEvidential:
    channels [4i, 4i+1, 4i+2, 4i+3] for target i = [γᵢ, νᵢ, αᵢ, βᵢ]
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn


class EvidentialRegressionLoss(nn.Module):
    """
    Masked evidential regression loss over NIG outputs.

    Args:
        lambda_coef : weight on the evidential regularizer L_R (default 0.1,
                      as used in the paper's U-Net depth experiment)
        weights     : per-target scalar weights; length must match num_targets
    """

    def __init__(
        self,
        lambda_coef: float = 0.1,
        weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.lambda_coef = float(lambda_coef)
        if weights is None:
            weights = [1.0, 1.0]
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred   : [B, N_t * 4, H, W]  – NIG params, channel layout [γ,ν,α,β] per target
            target : [B, N_t, H, W]       – ground-truth regression targets
            mask   : [B, H, W] bool        – True on valid pixels
        Returns:
            scalar loss
        """
        n_valid = mask.sum()
        if n_valid == 0:
            return pred.sum() * 0.0

        num_targets = target.shape[1]
        total_loss = torch.zeros(1, device=pred.device, dtype=pred.dtype)

        for i in range(num_targets):
            # Extract NIG parameters for target i
            gamma = pred[:, 4 * i + 0][mask]   # [N_valid]
            nu    = pred[:, 4 * i + 1][mask]   # [N_valid]  > 0
            alpha = pred[:, 4 * i + 2][mask]   # [N_valid]  > 1
            beta  = pred[:, 4 * i + 3][mask]   # [N_valid]  > 0
            y     = target[:, i][mask]          # [N_valid]

            loss_nll = _student_t_nll(y, gamma, nu, alpha, beta)
            loss_reg = _evidential_regularizer(y, gamma, nu, alpha)

            target_loss = loss_nll + self.lambda_coef * loss_reg
            total_loss = total_loss + self.weights[i] * target_loss

        return total_loss.squeeze()


def _student_t_nll(
    y: torch.Tensor,
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """
    Negative log-likelihood of the Student-t marginal likelihood (Eq. 8 / S1.2).

    L_NLL = 0.5·log(π/ν)
            − α·log(Ω)
            + (α + 0.5)·log((y − γ)²·ν + Ω)
            + lgamma(α) − lgamma(α + 0.5)

    where Ω = 2β(1 + ν).
    """
    two_beta_1pnu = 2.0 * beta * (1.0 + nu)   # Ω

    nll = (
        0.5 * torch.log(math.pi / nu)
        - alpha * torch.log(two_beta_1pnu)
        + (alpha + 0.5) * torch.log((y - gamma) ** 2 * nu + two_beta_1pnu)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    return nll.mean()


def _evidential_regularizer(
    y: torch.Tensor,
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """
    Evidential regularizer (Eq. 9): penalises high evidence on large errors.

    L_R = |y − γ| · (2ν + α)   where Φ = 2ν + α is the total evidence.
    """
    error = torch.abs(y - gamma)
    evidence = 2.0 * nu + alpha
    return (error * evidence).mean()


def build_evidential_loss(cfg: dict) -> EvidentialRegressionLoss:
    """Build EvidentialRegressionLoss from the full config dict."""
    train_cfg = cfg.get("training", {})
    evidential_cfg = train_cfg.get("evidential", {})
    lambda_coef = float(evidential_cfg.get("lambda_coef", 0.1))

    targets = cfg.get("data", {}).get("targets", ["tree_count", "mean_height"])
    lw = train_cfg.get("loss_weights", {})
    _default_weights = {"tree_count": 1.0, "mean_height": 1.0}
    weights = [float(lw.get(t, _default_weights.get(t, 1.0))) for t in targets]

    return EvidentialRegressionLoss(lambda_coef=lambda_coef, weights=weights)

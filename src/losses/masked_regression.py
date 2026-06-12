"""
Masked multi-target regression loss.

Only pixels where ``valid_mask`` is True contribute to the loss.
This is the central mechanism for handling sparse tree pixels.

Loss types  : mse | smooth_l1 | mae
Per-target weighting allows balancing tree_count vs mean_height.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedRegressionLoss(nn.Module):
    """
    Args:
        loss_type : 'mse' | 'smooth_l1' | 'mae'
        weights   : per-target scalar weights [tree_count_w, mean_height_w]
    """

    _SUPPORTED = ("mse", "smooth_l1", "mae")

    def __init__(
        self,
        loss_type: str = "mse",
        weights: List[float] | None = None,
    ) -> None:
        super().__init__()
        if loss_type not in self._SUPPORTED:
            raise ValueError(f"loss_type must be one of {self._SUPPORTED}, got '{loss_type}'")
        self.loss_type = loss_type

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
            pred   : [B, 2, H, W]
            target : [B, 2, H, W]
            mask   : [B, H, W] bool – True on valid (tree-present) pixels
        Returns:
            scalar loss
        """
        n_valid = mask.sum()
        if n_valid == 0:
            # Differentiable zero loss when no valid pixels in batch
            return pred.sum() * 0.0

        total_loss = torch.zeros(1, device=pred.device, dtype=pred.dtype)

        for i in range(pred.shape[1]):
            p = pred[:, i][mask]     # [N_valid]
            t = target[:, i][mask]   # [N_valid]

            if self.loss_type == "mse":
                loss_i = F.mse_loss(p, t)
            elif self.loss_type == "smooth_l1":
                loss_i = F.smooth_l1_loss(p, t)
            else:  # mae
                loss_i = F.l1_loss(p, t)

            total_loss = total_loss + self.weights[i] * loss_i

        return total_loss.squeeze()


def build_loss(cfg: dict) -> MaskedRegressionLoss:
    """Construct loss from the ``training`` sub-dict of the full config.

    Weights are assembled in the order of ``data.targets`` so the loss is
    correct for both single-task and dual-task configurations.
    """
    train_cfg = cfg.get("training", {})
    raw_type = train_cfg.get("loss", "masked_mse")
    loss_type = raw_type.replace("masked_", "")   # normalise 'masked_mse' → 'mse'

    targets = cfg.get("data", {}).get("targets", ["tree_count", "mean_height"])
    lw = train_cfg.get("loss_weights", {})
    # Default weight 1.0 per target; honour named weights when present
    _default_weights = {"tree_count": 1.0, "mean_height": 1.0}
    weights = [float(lw.get(t, _default_weights.get(t, 1.0))) for t in targets]

    return MaskedRegressionLoss(loss_type=loss_type, weights=weights)

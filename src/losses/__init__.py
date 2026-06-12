"""
Loss factory.

Dispatches on ``cfg["training"]["loss"]``:
  - ``"evidential_nig"``  → EvidentialRegressionLoss
  - anything else         → MaskedRegressionLoss  (masked_mse / masked_smooth_l1 / masked_mae)
"""

from __future__ import annotations

import torch.nn as nn


def build_loss(cfg: dict) -> nn.Module:
    loss_type = cfg.get("training", {}).get("loss", "masked_mse")
    if loss_type == "evidential_nig":
        from src.losses.evidential_regression import build_evidential_loss
        return build_evidential_loss(cfg)
    else:
        from src.losses.masked_regression import build_loss as _build_masked
        return _build_masked(cfg)

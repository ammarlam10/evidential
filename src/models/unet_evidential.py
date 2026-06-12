"""
UNet with ResNet50 encoder – evidential regression head.

Instead of outputting N_t channels (one per target), the final layer outputs
N_t * 4 channels representing the hyperparameters of a Normal Inverse-Gamma
(NIG) distribution per target: (γ, ν, α, β).

Channel layout: for target i, channels [4i, 4i+1, 4i+2, 4i+3] = [γ, ν, α, β].

Activations (Amini et al., NeurIPS 2020):
  γ  – linear (predicted mean)
  ν  – softplus (> 0)
  α  – softplus + 1 (> 1)
  β  – softplus (> 0)

At inference, γ = pred[:, 0::4] is the point prediction (E[µ]).
Uncertainty maps:
  aleatoric  ≈ β / (α − 1)       = pred[:, 3::4] / (pred[:, 2::4] − 1)
  epistemic  ≈ β / (ν(α − 1))    = aleatoric / pred[:, 1::4]
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from src.models.factory import register_model


@register_model("unet_evidential")
def build_unet_evidential(cfg: Dict, num_input_channels: int) -> nn.Module:
    unet_cfg = cfg.get("model", {}).get("unet", {})
    targets: List[str] = cfg.get("data", {}).get("targets", ["tree_count", "mean_height"])
    return UNetEvidential(
        in_channels=num_input_channels,
        encoder_name=unet_cfg.get("encoder_name", "resnet50"),
        encoder_weights=unet_cfg.get("encoder_weights", "imagenet"),
        decoder_channels=unet_cfg.get("decoder_channels", [256, 128, 64, 32, 16]),
        num_targets=len(targets),
    )


class UNetEvidential(nn.Module):
    """
    U-Net that outputs NIG hyperparameters (γ, ν, α, β) per target pixel.

    Args:
        in_channels      : number of input spectral channels
        encoder_name     : smp encoder key (e.g. 'resnet50')
        encoder_weights  : pretrained weights key (e.g. 'imagenet') or None
        decoder_channels : channel sizes for each decoder stage
        num_targets      : number of regression targets (N_t)

    Output shape: [B, N_t * 4, H, W]
        Channels [4i .. 4i+3] for target i = [γᵢ, νᵢ, αᵢ, βᵢ]
    """

    def __init__(
        self,
        in_channels: int,
        encoder_name: str = "resnet50",
        encoder_weights: Optional[str] = "imagenet",
        decoder_channels: Optional[list] = None,
        num_targets: int = 2,
    ) -> None:
        super().__init__()
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32, 16]

        self.num_targets = num_targets

        # smp.Unet final 1×1 conv produces num_targets * 4 raw channels
        self.backbone = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_targets * 4,
            activation=None,
            decoder_channels=decoder_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, N_t * 4, H, W]
                 channel layout: [γ₀, ν₀, α₀, β₀,  γ₁, ν₁, α₁, β₁, ...]
        """
        raw = self.backbone(x)          # [B, N_t*4, H, W], no activation

        # Apply NIG constraints per-group-of-4
        gamma = raw[:, 0::4]                     # linear  → E[µ]
        nu    = F.softplus(raw[:, 1::4])         # > 0
        alpha = F.softplus(raw[:, 2::4]) + 1.0  # > 1
        beta  = F.softplus(raw[:, 3::4])         # > 0

        # Interleave back: [B, N_t, 4, H, W] → [B, N_t*4, H, W]
        B, _, H, W = gamma.shape
        out = torch.stack([gamma, nu, alpha, beta], dim=2)   # [B, N_t, 4, H, W]
        return out.reshape(B, self.num_targets * 4, H, W)

"""
Spatial augmentations that operate jointly on (x, y, valid_mask) tensors
so that input patches, target maps, and validity masks stay aligned.

All transforms operate on torch.Tensors with shapes:
  x    : [C, H, W]  float32
  y    : [2, H, W]  float32
  mask : [H, W]     bool

Usage:
    from src.data.transforms import build_train_transform
    transform = build_train_transform()
    x, y, mask = transform(x, y, mask)
"""

from __future__ import annotations

import random
from typing import List, Tuple

import torch


# ── primitives ────────────────────────────────────────────────────────────────

class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            x = torch.flip(x, dims=[-1])
            y = torch.flip(y, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        return x, y, mask


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            x = torch.flip(x, dims=[-2])
            y = torch.flip(y, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        return x, y, mask


class RandomRotate90:
    """Randomly rotate by 0 / 90 / 180 / 270 degrees."""

    def __init__(self, p: float = 0.75) -> None:
        self.p = p

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            k = random.randint(1, 3)
            x = torch.rot90(x, k, dims=[-2, -1])
            y = torch.rot90(y, k, dims=[-2, -1])
            mask = torch.rot90(mask, k, dims=[-2, -1])
        return x, y, mask


# ── composition ───────────────────────────────────────────────────────────────

class Compose:
    def __init__(self, transforms: List) -> None:
        self.transforms = transforms

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            x, y, mask = t(x, y, mask)
        return x, y, mask


# ── public builders ───────────────────────────────────────────────────────────

def build_train_transform() -> Compose:
    return Compose(
        [
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotate90(p=0.75),
        ]
    )


def build_val_transform() -> None:
    """No spatial augmentation at validation/test time."""
    return None

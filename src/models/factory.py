"""
Model registry and factory.

Adding a new model:
  1. Create src/models/my_model.py
  2. Decorate the builder function with @register_model("my_key")
  3. Import the module in src/models/__init__.py

The builder signature must be:
    def build_fn(cfg: dict, num_input_channels: int) -> torch.nn.Module

The returned module must satisfy:
    forward(x: Tensor[B, C, H, W]) -> Tensor[B, N_t * 4, H, W]  (evidential NIG head)
where N_t is the number of regression targets and each target has 4 channels (γ, ν, α, β).
"""

from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

_REGISTRY: Dict[str, Callable] = {}


def register_model(name: str) -> Callable:
    """Class/function decorator that registers a model builder."""

    def decorator(fn: Callable) -> Callable:
        if name in _REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        _REGISTRY[name] = fn
        return fn

    return decorator


def build_model(cfg: Dict, num_input_channels: int) -> nn.Module:
    """
    Instantiate and return a model from the registry.

    Args:
        cfg               : full config dict (``model`` sub-dict is consumed)
        num_input_channels: number of input feature channels
    """
    model_cfg = cfg.get("model", cfg)
    name = model_cfg.get("name", "unet_evidential")
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Registered models: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](cfg, num_input_channels)


def list_models() -> list:
    return sorted(_REGISTRY.keys())

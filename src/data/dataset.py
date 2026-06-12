"""
BiomassDataset – Zarr-backed patch dataset for pixel-wise regression.

Input channel layout (deterministic, matches norm_stats channel order):
  [0:8]   S1 – 4 seasons × 2 bands (VV, VH)
  [8:48]  S2 – 4 seasons × 10 bands (B2 B3 B4 B8 B5 B6 B7 B8A B11 B12)
  [48]    tree_species (optional) – categorical class index per pixel, not spectral
          data; compute_stats uses identity norm (mean 0, std 1) so values are
          unchanged after ``(x - mean) / std``.

Targets (configurable via ``data.targets``):
  Default: ["tree_count", "mean_height"]  → y shape [2, 128, 128]
  Single-task examples:
    ["mean_height"]  → y shape [1, 128, 128]
    ["tree_count"]   → y shape [1, 128, 128]

  Both zarr arrays are always loaded to build valid_mask (mask is driven by
  isfinite(mean_height) regardless of which targets are requested).

Returns (x, y, valid_mask):
  x          – float32 tensor [C, 128, 128], normalised, NaN→0
  y          – float32 tensor [N_t, 128, 128], fill NaN→0 (masked in loss)
  valid_mask – bool tensor    [128, 128]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset

# ── canonical band/season orders ─────────────────────────────────────────────
S1_SEASONS: List[str] = ["summer", "autumn", "spring", "winter"]
S2_SEASONS: List[str] = ["summer", "autumn", "spring", "winter"]
S1_BANDS: List[str] = ["VV", "VH"]
S2_BANDS: List[str] = ["B2", "B3", "B4", "B8", "B5", "B6", "B7", "B8A", "B11", "B12"]

# ── supported target names (order defines channel index in the dual-task case) ─
ALL_TARGETS: List[str] = ["tree_count", "mean_height"]


def build_channel_names(
    s1_seasons: List[str],
    s2_seasons: List[str],
    use_species: bool,
) -> List[str]:
    names: List[str] = []
    for season in s1_seasons:
        for band in S1_BANDS:
            names.append(f"s1_{season}_{band}")
    for season in s2_seasons:
        for band in S2_BANDS:
            names.append(f"s2_{season}_{band}")
    if use_species:
        names.append("tree_species")
    return names


class BiomassDataset(Dataset):
    """
    Args:
        root        : dataset root directory (inside container: /data)
        split       : one of 'train', 'val', 'test'
        cfg         : full config dict (the ``data`` sub-dict is consumed)
        norm_stats  : dict with keys 'mean' and 'std' (list[float], per channel)
                      obtained from scripts/compute_stats.py; pass None to skip
        transform   : optional callable(x, y, mask) → (x, y, mask)
    """

    def __init__(
        self,
        root: str,
        split: str,
        cfg: Dict,
        norm_stats: Optional[Dict] = None,
        transform=None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.cfg = cfg
        self.norm_stats = norm_stats
        self.transform = transform

        data_cfg = cfg.get("data", cfg)  # accept both full cfg and data sub-dict
        self.s1_seasons: List[str] = data_cfg.get("s1_seasons", S1_SEASONS)
        self.s2_seasons: List[str] = data_cfg.get("s2_seasons", S2_SEASONS)
        self.use_species: bool = data_cfg.get("use_species", True)
        self.valid_mask_mode: str = data_cfg.get("valid_mask_mode", "notnull")
        self.target_transform: Dict = data_cfg.get("target_transform", {})

        # Which targets to include in y. Both zarr arrays are always opened to
        # build valid_mask; only the requested subset is stacked into y.
        self.targets: List[str] = list(data_cfg.get("targets", ALL_TARGETS))
        for t in self.targets:
            if t not in ALL_TARGETS:
                raise ValueError(
                    f"Unknown target '{t}'. Supported: {ALL_TARGETS}"
                )

        # ── load patch indices for this split ─────────────────────────────────
        split_file = data_cfg.get("split_file", str(self.root / "patch_index_subset.parquet"))
        split_col = data_cfg.get("split_column", "split")
        idx_col = data_cfg.get("patch_idx_column", "patch_idx")

        df = pd.read_parquet(split_file)
        self._validate_parquet_columns(df, split_col, idx_col)
        mask_split = df[split_col] == split
        if mask_split.sum() == 0:
            raise ValueError(
                f"No rows found for split='{split}' in column '{split_col}'. "
                f"Available values: {df[split_col].unique().tolist()}"
            )
        self.patch_indices: np.ndarray = df.loc[mask_split, idx_col].to_numpy(dtype=np.int64)

        # ── open zarr stores lazily (no data loaded yet) ──────────────────────
        self._s1 = {
            season: zarr.open(str(self.root / "inputs" / f"s1_{season}"), mode="r")
            for season in self.s1_seasons
        }
        self._s2 = {
            season: zarr.open(str(self.root / "inputs" / f"s2_{season}"), mode="r")
            for season in self.s2_seasons
        }
        if self.use_species:
            self._species = zarr.open(str(self.root / "inputs" / "tree_species"), mode="r")
        else:
            self._species = None

        self._tree_count = zarr.open(str(self.root / "labels" / "tree_count"), mode="r")
        self._mean_height = zarr.open(str(self.root / "labels" / "mean_height"), mode="r")

        # ── channel registry ──────────────────────────────────────────────────
        self.channel_names: List[str] = build_channel_names(
            self.s1_seasons, self.s2_seasons, self.use_species
        )

        # ── pre-compute normalisation tensors (broadcast-ready) ───────────────
        if norm_stats is not None:
            mean = np.array(norm_stats["mean"], dtype=np.float32)
            std = np.array(norm_stats["std"], dtype=np.float32)
            std = np.where(std < 1e-8, 1.0, std)
            self._norm_mean = mean[:, None, None]   # [C, 1, 1]
            self._norm_std = std[:, None, None]
        else:
            self._norm_mean = None
            self._norm_std = None

    # ── public interface ──────────────────────────────────────────────────────

    @property
    def num_channels(self) -> int:
        return len(self.channel_names)

    def __len__(self) -> int:
        return len(self.patch_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patch_idx = int(self.patch_indices[idx])

        x = self._load_input(patch_idx)          # [C, H, W] float32
        tc, mh, valid_mask = self._load_targets(patch_idx)  # [H,W], [H,W], [H,W] bool

        # ── normalise input ───────────────────────────────────────────────────
        if self._norm_mean is not None:
            x = (x - self._norm_mean) / self._norm_std

        # NaN in input (missing acquisition) → 0 after normalisation
        np.nan_to_num(x, nan=0.0, copy=False)

        # ── target transforms ─────────────────────────────────────────────────
        tc_safe = np.where(np.isfinite(tc), tc, 0.0)
        mh_safe = np.where(np.isfinite(mh), mh, 0.0)

        if self.target_transform.get("tree_count") == "log1p":
            tc_safe = np.log1p(np.maximum(tc_safe, 0.0))
        if self.target_transform.get("mean_height") == "log1p":
            mh_safe = np.log1p(np.maximum(mh_safe, 0.0))

        # Stack only the requested target channels into y [N_t, H, W]
        _target_arrays = {"tree_count": tc_safe, "mean_height": mh_safe}
        y = np.stack(
            [_target_arrays[t] for t in self.targets], axis=0
        ).astype(np.float32)

        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)
        mask_t = torch.from_numpy(valid_mask)

        if self.transform is not None:
            x_t, y_t, mask_t = self.transform(x_t, y_t, mask_t)

        return x_t, y_t, mask_t

    # ── private helpers ───────────────────────────────────────────────────────

    def _load_input(self, patch_idx: int) -> np.ndarray:
        """Returns float32 array [C, H, W]."""
        channels = []

        # S1: zarr shape [N, H, W, 2] → index gives [H, W, 2] → transpose [2, H, W]
        for season in self.s1_seasons:
            arr = np.array(self._s1[season][patch_idx], dtype=np.float32)  # [H, W, 2]
            channels.append(arr.transpose(2, 0, 1))

        # S2: zarr shape [N, H, W, 10] → [10, H, W]
        for season in self.s2_seasons:
            arr = np.array(self._s2[season][patch_idx], dtype=np.float32)  # [H, W, 10]
            channels.append(arr.transpose(2, 0, 1))

        # Species: zarr shape [N, H, W] → [H, W] → [1, H, W]
        if self._species is not None:
            arr = np.array(self._species[patch_idx], dtype=np.float32)     # [H, W]
            channels.append(arr[np.newaxis])

        return np.concatenate(channels, axis=0)  # [C, H, W]

    def _load_targets(
        self, patch_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns tree_count [H,W], mean_height [H,W], valid_mask [H,W] bool."""
        tc = np.array(self._tree_count[patch_idx], dtype=np.float32)   # [H, W]
        mh = np.array(self._mean_height[patch_idx], dtype=np.float32)  # [H, W]

        if self.valid_mask_mode == "positive":
            valid = np.isfinite(tc) & np.isfinite(mh) & (tc > 0)
        else:
            valid = np.isfinite(tc) & np.isfinite(mh)

        return tc, mh, valid.astype(bool)

    @staticmethod
    def _validate_parquet_columns(df: pd.DataFrame, split_col: str, idx_col: str) -> None:
        missing = [c for c in [split_col, idx_col] if c not in df.columns]
        if missing:
            raise KeyError(
                f"Expected columns {missing} not found in parquet file.\n"
                f"Available columns: {df.columns.tolist()}\n"
                "Update data.split_column / data.patch_idx_column in configs/unet_resnet50.yaml "
                "or configs/xgboost.yaml."
            )

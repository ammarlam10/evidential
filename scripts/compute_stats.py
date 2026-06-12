"""
Compute per-channel mean and std over the training split and save to JSON.

Run this ONCE before training:
    python scripts/compute_stats.py --config configs/experiments/singletask_height_unet_evidential.yaml

Parallelism:
    --num_workers N  splits the patch list into N chunks and computes partial
    (sum, sum_sq, count) per chunk concurrently via ProcessPoolExecutor.
    Each worker opens its own Zarr read handles, so there is no shared state.
    Partial results are merged via the standard online formula:
        mean  = sum / count
        var   = sum_sq / count - mean^2
    Recommended: --num_workers 12 on the 48-vCPU host.

Optional --subsample (0,1] to use a fraction of train patches for speed.
Pass --inspect to print parquet columns without computing stats.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import build_channel_names
from src.utils.config import load_config, save_norm_stats


# ── per-worker function (must be top-level for pickling) ─────────────────────

def _worker_stats(
    patch_indices: np.ndarray,
    root: str,
    s1_seasons: List[str],
    s2_seasons: List[str],
    use_species: bool,
    n_channels: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute partial (sum_x, sum_x2, count) for a chunk of patch indices.
    Each worker opens its own Zarr stores to avoid cross-process contention.
    Returns three float64 arrays of shape [n_channels].
    """
    root_path = Path(root)

    s1_stores = {
        season: zarr.open(str(root_path / "inputs" / f"s1_{season}"), mode="r")
        for season in s1_seasons
    }
    s2_stores = {
        season: zarr.open(str(root_path / "inputs" / f"s2_{season}"), mode="r")
        for season in s2_seasons
    }
    species_store = None
    if use_species:
        species_store = zarr.open(str(root_path / "inputs" / "tree_species"), mode="r")

    sum_x = np.zeros(n_channels, dtype=np.float64)
    sum_x2 = np.zeros(n_channels, dtype=np.float64)
    count = np.zeros(n_channels, dtype=np.int64)

    for patch_idx in patch_indices:
        c = 0
        for season in s1_seasons:
            arr = np.array(s1_stores[season][patch_idx], dtype=np.float32)  # [H, W, 2]
            for b in range(arr.shape[-1]):
                ch = arr[:, :, b]
                valid = np.isfinite(ch)
                if valid.any():
                    vals = ch[valid].astype(np.float64)
                    sum_x[c] += vals.sum()
                    sum_x2[c] += (vals ** 2).sum()
                    count[c] += valid.sum()
                c += 1

        for season in s2_seasons:
            arr = np.array(s2_stores[season][patch_idx], dtype=np.float32)  # [H, W, 10]
            for b in range(arr.shape[-1]):
                ch = arr[:, :, b]
                valid = np.isfinite(ch)
                if valid.any():
                    vals = ch[valid].astype(np.float64)
                    sum_x[c] += vals.sum()
                    sum_x2[c] += (vals ** 2).sum()
                    count[c] += valid.sum()
                c += 1

        if species_store is not None:
            arr = np.array(species_store[patch_idx], dtype=np.float32)  # [H, W]
            valid = np.isfinite(arr)
            if valid.any():
                vals = arr[valid].astype(np.float64)
                sum_x[c] += vals.sum()
                sum_x2[c] += (vals ** 2).sum()
                count[c] += valid.sum()

    return sum_x, sum_x2, count


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute per-channel normalisation stats")
    p.add_argument("--config", default="configs/unet_resnet50.yaml")
    p.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="Fraction of train patches to use (0, 1]. Default: 1.0 (all).",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=max(1, mp.cpu_count() // 2),
        help=(
            "Number of parallel processes for stats accumulation. "
            "Default: half the CPU count. Recommended: 12 on the 48-vCPU host."
        ),
    )
    p.add_argument(
        "--inspect",
        action="store_true",
        help="Print parquet columns and exit without computing stats.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]

    root = str(data_cfg["root"])
    split_file = data_cfg.get("split_file", str(Path(root) / "patch_index_subset.parquet"))
    split_col = data_cfg.get("split_column", "split")
    idx_col = data_cfg.get("patch_idx_column", "patch_idx")

    df = pd.read_parquet(split_file)
    if args.inspect:
        print("Parquet columns:", df.columns.tolist())
        print("Sample rows:")
        print(df.head())
        return

    if split_col not in df.columns or idx_col not in df.columns:
        print(
            f"[ERROR] Expected columns '{split_col}' and '{idx_col}' not found.\n"
            f"Available: {df.columns.tolist()}\n"
            "Use --inspect to explore the file, then update configs/unet_resnet50.yaml."
        )
        sys.exit(1)

    train_indices = df.loc[df[split_col] == "train", idx_col].to_numpy(dtype=np.int64)
    print(f"Found {len(train_indices)} training patches.")

    if args.subsample < 1.0:
        rng = np.random.default_rng(42)
        n = max(1, int(len(train_indices) * args.subsample))
        train_indices = rng.choice(train_indices, size=n, replace=False)
        print(f"Subsampled to {n} patches ({args.subsample*100:.0f}%).")

    s1_seasons = data_cfg.get("s1_seasons", ["summer", "autumn", "spring", "winter"])
    s2_seasons = data_cfg.get("s2_seasons", ["summer", "autumn", "spring", "winter"])
    use_species = data_cfg.get("use_species", True)

    channel_names = build_channel_names(s1_seasons, s2_seasons, use_species)
    C = len(channel_names)
    print(f"Computing stats for {C} channels using {args.num_workers} workers …")

    # ── split into chunks, one per worker ─────────────────────────────────────
    num_workers = min(args.num_workers, len(train_indices))
    chunks = np.array_split(train_indices, num_workers)

    global_sum = np.zeros(C, dtype=np.float64)
    global_sum2 = np.zeros(C, dtype=np.float64)
    global_count = np.zeros(C, dtype=np.int64)

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(
                _worker_stats, chunk, root, s1_seasons, s2_seasons, use_species, C
            ): i
            for i, chunk in enumerate(chunks)
        }
        pbar = tqdm(total=len(train_indices), desc="Computing stats")
        for future in as_completed(futures):
            chunk_idx = futures[future]
            s, s2, cnt = future.result()
            global_sum += s
            global_sum2 += s2
            global_count += cnt
            pbar.update(len(chunks[chunk_idx]))
        pbar.close()

    # ── merge partial results → mean / std ───────────────────────────────────
    denom = np.maximum(global_count, 1).astype(np.float64)
    mean = global_sum / denom
    var = global_sum2 / denom - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-8))

    # tree_species holds categorical class IDs, not continuous reflectance. Z-scoring
    # would wrongly treat codewords as an ordinal scale. Use identity norm so the
    # network receives raw integer codes (same as stored in zarr).
    species_idx = [i for i, n in enumerate(channel_names) if n == "tree_species"]
    if species_idx:
        i = species_idx[0]
        mean[i] = 0.0
        std[i] = 1.0

    stats = {
        "channel_names": channel_names,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": global_count.tolist(),
        "n_patches": int(len(train_indices)),
    }

    out_path = data_cfg.get("norm_stats_path", "/workspace/artifacts/norm_stats.json")
    save_norm_stats(stats, out_path)
    print(f"\nNorm stats saved → {out_path}")

    print(f"  S1 mean range : {mean[:8].min():.4f} – {mean[:8].max():.4f}")
    print(f"  S2 mean range : {mean[8:48].min():.4f} – {mean[8:48].max():.4f}")
    if use_species and species_idx:
        print("  species       : identity norm (class IDs, not z-scored)")


if __name__ == "__main__":
    main()

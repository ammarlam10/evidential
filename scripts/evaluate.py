"""
Evaluate a saved checkpoint on val or test split.

Usage (inside container):
    python scripts/evaluate.py \\
        --config configs/experiments/singletask_height_unet_evidential.yaml \\
        --checkpoint /workspace/artifacts/checkpoints/best.pt \\
        --split test \\
        --output /workspace/artifacts/test_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.models  # noqa: F401 – triggers @register_model decorators
from src.data.dataset import BiomassDataset
from src.losses import build_loss
from src.losses.evidential_regression import EvidentialRegressionLoss
from src.models.factory import build_model
from src.training.metrics import compute_masked_metrics, get_inverse_transforms
from src.utils.config import load_config, load_norm_stats


def _point_pred(pred: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Extract γ (mean) channels from evidential NIG output if applicable."""
    if isinstance(build_loss(cfg), EvidentialRegressionLoss):
        return pred[:, 0::4]
    return pred


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate biomass regression model")
    p.add_argument("--config", default="configs/experiments/singletask_height_unet_evidential.yaml")
    p.add_argument(
        "--checkpoint",
        default="/workspace/artifacts/checkpoints/best.pt",
    )
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument(
        "--output",
        default=None,
        help="Path to write metrics JSON (default: <ckpt_dir>/<split>_metrics.json)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── load config (prefer checkpoint's embedded cfg for reproducibility) ────
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("cfg") or load_config(args.config)
    print(f"Evaluating checkpoint from epoch {ckpt.get('epoch', '?')}")

    # ── norm stats ────────────────────────────────────────────────────────────
    norm_stats_path = cfg["data"].get("norm_stats_path", "/workspace/artifacts/norm_stats.json")
    norm_stats = load_norm_stats(norm_stats_path)

    # ── dataset / loader ──────────────────────────────────────────────────────
    train_cfg = cfg.get("training", {})
    ds = BiomassDataset(
        root=cfg["data"]["root"],
        split=args.split,
        cfg=cfg,
        norm_stats=norm_stats,
        transform=None,
    )
    loader = DataLoader(
        ds,
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=False,
    )
    print(f"Split '{args.split}': {len(ds)} patches")

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg, num_input_channels=ds.num_channels)
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    inv_transforms = get_inverse_transforms(cfg)

    # ── inference ─────────────────────────────────────────────────────────────
    all_preds, all_targets, all_masks = [], [], []
    total_support = 0.0

    with torch.no_grad():
        for x, y, mask in tqdm(loader, desc=f"Eval [{args.split}]"):
            x = x.to(device, non_blocking=True)
            pred = _point_pred(model(x), cfg)
            all_preds.append(pred.cpu())
            all_targets.append(y)
            all_masks.append(mask)
            total_support += mask.float().mean().item()

    all_preds_t = torch.cat(all_preds, dim=0)
    all_targets_t = torch.cat(all_targets, dim=0)
    all_masks_t = torch.cat(all_masks, dim=0)

    metrics = compute_masked_metrics(all_preds_t, all_targets_t, all_masks_t, inv_transforms)
    metrics["split"] = args.split
    metrics["n_patches"] = len(ds)
    metrics["avg_support_ratio"] = total_support / max(len(loader), 1)
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["epoch"] = int(ckpt.get("epoch", -1))

    # ── print & save ──────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  Split          : {args.split}")
    print(f"  Patches        : {metrics['n_patches']}")
    print(f"  Support ratio  : {metrics['avg_support_ratio']:.4f}")
    print(f"  RMSE tree_count: {metrics.get('rmse_tree_count', float('nan')):.4f}")
    print(f"  MAE  tree_count: {metrics.get('mae_tree_count', float('nan')):.4f}")
    print(f"  R2   tree_count: {metrics.get('r2_tree_count', float('nan')):.4f}")
    print(f"  RMSE mean_height: {metrics.get('rmse_mean_height', float('nan')):.4f}")
    print(f"  MAE  mean_height: {metrics.get('mae_mean_height', float('nan')):.4f}")
    print(f"  R2   mean_height: {metrics.get('r2_mean_height', float('nan')):.4f}")
    if "rmse_tree_count_orig" in metrics:
        print(f"  RMSE tree_count (original scale): {metrics['rmse_tree_count_orig']:.4f}")
        print(f"  RMSE mean_height (original scale): {metrics.get('rmse_mean_height_orig', float('nan')):.4f}")
    print(f"{'─'*55}\n")

    out_path = args.output or str(
        Path(args.checkpoint).parent / f"{args.split}_metrics.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({k: (v if v == v else None) for k, v in metrics.items()}, fh, indent=2)
    print(f"Metrics saved → {out_path}")


if __name__ == "__main__":
    main()

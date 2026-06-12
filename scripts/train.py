"""
Main training entrypoint – single-GPU and multi-GPU (DDP).

Single-GPU (default):
    python scripts/train.py --config configs/experiments/singletask_height_unet_evidential.yaml

Multi-GPU DDP via torchrun (e.g. 2 GPUs):
    torchrun --standalone --nproc_per_node=2 \\
        scripts/train.py --config configs/experiments/singletask_height_unet_evidential.yaml --ddp

Or use the docker-compose `train_ddp` service which sets this up automatically.

Run compute_stats.py first to generate norm_stats.json.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.models  # noqa: F401 – triggers @register_model decorators
from src.data.dataset import BiomassDataset
from src.data.transforms import build_train_transform, build_val_transform
from src.models.factory import build_model
from src.training.trainer import Trainer
from src.utils.config import load_config, load_norm_stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train biomass regression model")
    p.add_argument("--config", default="configs/experiments/singletask_height_unet_evidential.yaml")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument(
        "--ddp",
        action="store_true",
        help=(
            "Enable DistributedDataParallel. Must be launched via torchrun. "
            "LOCAL_RANK is read from env automatically."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override training.seed from config",
    )
    p.add_argument(
        "--run-dir",
        dest="run_dir",
        default=None,
        help=(
            "Root directory for this run's artifacts. "
            "Overrides training.save_dir, logging.tensorboard_dir, and logging.csv_path "
            "to <run_dir>/checkpoints, <run_dir>/runs, and <run_dir>/metrics.csv respectively."
        ),
    )
    p.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        default=None,
        help="Override training.num_workers (lower this when many parallel jobs exhaust /dev/shm in Docker).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs from config.",
    )
    p.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=None,
        help="Override training.batch_size from config.",
    )
    p.add_argument(
        "--validate-every",
        dest="validate_every_n_epochs",
        type=int,
        default=None,
        help="Override training.validate_every_n_epochs (val run every N epochs; 1 = every epoch).",
    )
    p.add_argument(
        "--early-stopping-patience",
        dest="early_stopping_patience",
        type=int,
        default=None,
        help=(
            "Override training.early_stopping_patience (epochs without val improvement); "
            "0 disables early stopping."
        ),
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override training.lr from config.",
    )
    freeze_group = p.add_mutually_exclusive_group()
    freeze_group.add_argument(
        "--freeze-clay-encoder",
        dest="freeze_clay_encoder",
        action="store_true",
        help="Set model.clay.freeze_encoder=true (Phase 1 Clay training).",
    )
    freeze_group.add_argument(
        "--no-freeze-clay-encoder",
        dest="freeze_clay_encoder",
        action="store_false",
        help="Set model.clay.freeze_encoder=false (Phase 2 Clay training).",
    )
    # Default None means "do not override config"; only changed when flag is explicitly passed.
    p.set_defaults(freeze_clay_encoder=None)
    return p.parse_args()


def set_seed(seed: int, deterministic: bool = False, cudnn_benchmark: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # benchmark=True selects the fastest conv algorithm for fixed-size inputs.
        # Provides measurable speedup on V100 with 128×128 patches.
        torch.backends.cudnn.benchmark = cudnn_benchmark


def build_loader(
    dataset: BiomassDataset,
    cfg: dict,
    shuffle: bool,
    sampler=None,
) -> DataLoader:
    train_cfg = cfg.get("training", {})
    num_workers = train_cfg.get("num_workers", 12)
    prefetch_factor = train_cfg.get("prefetch_factor", 2)
    return DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=train_cfg.get("pin_memory", True) and torch.cuda.is_available(),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # ── CLI overrides (used by run_ensemble.py for multi-seed parallel runs) ──
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    if args.run_dir is not None:
        cfg["training"]["save_dir"] = os.path.join(args.run_dir, "checkpoints")
        cfg["logging"]["tensorboard_dir"] = os.path.join(args.run_dir, "runs")
        cfg["logging"]["csv_path"] = os.path.join(args.run_dir, "metrics.csv")
    if args.num_workers is not None:
        cfg["training"]["num_workers"] = args.num_workers
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.validate_every_n_epochs is not None:
        cfg["training"]["validate_every_n_epochs"] = args.validate_every_n_epochs
    if args.early_stopping_patience is not None:
        cfg["training"]["early_stopping_patience"] = args.early_stopping_patience
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.freeze_clay_encoder is not None:
        cfg.setdefault("model", {}).setdefault("clay", {})["freeze_encoder"] = args.freeze_clay_encoder

    train_cfg = cfg.get("training", {})

    # ── DDP initialisation ────────────────────────────────────────────────────
    local_rank = -1
    world_size = 1
    if args.ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        world_size = dist.get_world_size()
        if local_rank == 0:
            print(f"DDP enabled: {world_size} processes")

    is_main = local_rank in (-1, 0)

    set_seed(
        train_cfg.get("seed", 42),
        train_cfg.get("deterministic", False),
        train_cfg.get("cudnn_benchmark", True),
    )

    # ── norm stats ────────────────────────────────────────────────────────────
    norm_stats_path = cfg["data"].get("norm_stats_path", "/workspace/artifacts/norm_stats.json")
    if not Path(norm_stats_path).exists():
        if is_main:
            print(
                f"[ERROR] Norm stats not found at '{norm_stats_path}'.\n"
                "Run compute_stats.py first:\n"
                "    python scripts/compute_stats.py --config configs/experiments/singletask_height_unet_evidential.yaml"
            )
        sys.exit(1)
    norm_stats = load_norm_stats(norm_stats_path)
    if is_main:
        print(f"Loaded norm stats from {norm_stats_path}  ({len(norm_stats['mean'])} channels)")

    # ── datasets ──────────────────────────────────────────────────────────────
    data_root = cfg["data"]["root"]
    train_ds = BiomassDataset(
        root=data_root, split="train", cfg=cfg,
        norm_stats=norm_stats, transform=build_train_transform(),
    )
    val_ds = BiomassDataset(
        root=data_root, split="val", cfg=cfg,
        norm_stats=norm_stats, transform=build_val_transform(),
    )
    test_ds = BiomassDataset(
        root=data_root, split="test", cfg=cfg,
        norm_stats=norm_stats, transform=build_val_transform(),
    )
    if is_main:
        print(
            f"Train: {len(train_ds)} patches  |  "
            f"Val: {len(val_ds)} patches  |  "
            f"Test: {len(test_ds)} patches"
        )
        print(f"Input channels: {train_ds.num_channels}")

    # ── samplers (DDP requires DistributedSampler for even shard distribution) ─
    # Test evaluation runs on rank 0 only with the unwrapped model, so no
    # DistributedSampler is needed for the test split.
    train_sampler = None
    val_sampler = None
    if args.ddp:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=local_rank, shuffle=False)

    train_loader = build_loader(train_ds, cfg, shuffle=True, sampler=train_sampler)
    val_loader = build_loader(val_ds, cfg, shuffle=False, sampler=val_sampler)
    test_loader = build_loader(test_ds, cfg, shuffle=False, sampler=None)

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg, num_input_channels=train_ds.num_channels)
    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {cfg['model']['name']}  |  Trainable params: {n_params:,}")

    # ── (optional) resume ─────────────────────────────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        if is_main:
            print(f"Resumed from epoch {ckpt['epoch']}  [{args.resume}]")

    # ── train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(model, cfg, local_rank=local_rank)
    trainer.fit(train_loader, val_loader, test_loader=test_loader)

    if args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""
Trainer: epoch loop, checkpointing, and TensorBoard/CSV logging.

Supports:
  - Single-GPU training (default)
  - DistributedDataParallel (DDP) via torchrun: pass local_rank >= 0
  - Mixed precision (AMP)
  - Gradient accumulation
  - Streaming validation metrics (RunningMetrics – no end-of-epoch cat)
  - Optional sparse validation (``validate_every_n_epochs``) and early stopping
    on combined val RMSE (``early_stopping_patience`` training epochs without improvement)

Model contract: forward(x: [B,C,H,W]) → [B,2,H,W]
DataLoader contract: yields (x, y, valid_mask)

This class is intentionally model-agnostic; works unchanged for UNet,
ViT, Prithvi, etc.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.losses import build_loss
from src.losses.evidential_regression import EvidentialRegressionLoss
from src.losses.masked_regression import MaskedRegressionLoss
from src.training.metrics import RunningMetrics, get_inverse_transforms, TARGET_NAMES


class Trainer:
    """
    Args:
        model      : nn.Module with forward(x) → [B, 2, H, W]
        cfg        : full config dict
        local_rank : DDP local rank (0-based). -1 = single-GPU / CPU.
        save_dir   : checkpoint dir (overrides cfg if given)
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Dict,
        local_rank: int = -1,
        save_dir: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        train_cfg = cfg.get("training", {})
        log_cfg = cfg.get("logging", {})

        self.epochs: int = train_cfg.get("epochs", 100)
        self.grad_accum: int = max(1, train_cfg.get("gradient_accumulation_steps", 1))
        self.log_every: int = log_cfg.get("log_every_n_steps", 10)

        # Validation frequency (1 = every epoch). Checkpoint selection uses val metrics.
        # Default: every 5 epochs (skip val on other epochs for speed).
        ve = int(train_cfg.get("validate_every_n_epochs", 5))
        self.validate_every_n_epochs: int = max(1, ve)

        # Early stopping: stop if no val improvement for this many *training* epochs
        # since the last best val checkpoint. None / 0 disables.
        # Default: 10 epochs for paper / experiment runs.
        esp = train_cfg.get("early_stopping_patience", 10)
        self.early_stopping_patience: Optional[int] = (
            None if esp is None or int(esp) <= 0 else int(esp)
        )
        self._best_epoch_for_es: Optional[int] = None  # epoch of last val improvement

        # ── device & DDP ──────────────────────────────────────────────────────
        self.ddp: bool = local_rank >= 0 and dist.is_available() and dist.is_initialized()
        if self.ddp:
            self.device = torch.device(f"cuda:{local_rank}")
            self.local_rank = local_rank
            self.is_main = local_rank == 0
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.local_rank = -1
            self.is_main = True

        self.amp: bool = train_cfg.get("amp", True) and self.device.type == "cuda"

        model = model.to(self.device)
        if self.ddp:
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        self.model = model

        # ── loss ──────────────────────────────────────────────────────────────
        self.criterion: nn.Module = build_loss(cfg)
        self._is_evidential: bool = isinstance(self.criterion, EvidentialRegressionLoss)

        # ── optimiser ─────────────────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg.get("lr", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )

        # ── AMP scaler ────────────────────────────────────────────────────────
        self.scaler: Optional[GradScaler] = GradScaler() if self.amp else None

        # ── scheduler ─────────────────────────────────────────────────────────
        self._scheduler_name: str = train_cfg.get("scheduler", "cosine")
        self._scheduler = None

        # ── active targets (single- or multi-task) ────────────────────────────
        self._target_names: list = list(
            cfg.get("data", {}).get("targets", TARGET_NAMES)
        )

        # ── streaming metrics ─────────────────────────────────────────────────
        inv_transforms = get_inverse_transforms(cfg, self._target_names)
        self._train_rm = RunningMetrics(inv_transforms, target_names=self._target_names)
        self._val_rm = RunningMetrics(inv_transforms, target_names=self._target_names)

        # ── checkpointing ─────────────────────────────────────────────────────
        ckpt_dir = save_dir or train_cfg.get("save_dir", "/workspace/artifacts/checkpoints")
        self.save_dir = Path(ckpt_dir)
        if self.is_main:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # ── logging (main process only) ───────────────────────────────────────
        if self.is_main:
            tb_dir = log_cfg.get("tensorboard_dir", "/workspace/artifacts/runs")
            self.writer = SummaryWriter(tb_dir)
            csv_path = log_cfg.get("csv_path", "/workspace/artifacts/metrics.csv")
            self.csv_path = Path(csv_path)
        else:
            self.writer = None
            self.csv_path = None

        self._csv_file = None
        self._csv_writer = None
        self.best_val_rmse: float = float("inf")
        self.global_step: int = 0

    # ── public interface ──────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ) -> None:
        self._build_scheduler(train_loader)
        if self.is_main:
            self._open_csv()
        t0 = time.time()

        try:
            for epoch in range(1, self.epochs + 1):
                if self.ddp:
                    # Ensures each process sees a different shuffle ordering
                    train_loader.sampler.set_epoch(epoch)

                train_metrics = self._run_epoch(train_loader, epoch, training=True)

                run_val = (
                    epoch % self.validate_every_n_epochs == 0
                    or epoch == self.epochs
                )
                val_metrics: Optional[Dict[str, float]] = None
                if run_val:
                    val_metrics = self._run_epoch(val_loader, epoch, training=False)

                should_stop = False
                if self.is_main:
                    ckpt_metrics: Dict[str, Any] = (
                        dict(val_metrics) if val_metrics else dict(train_metrics)
                    )
                    self._save_checkpoint("latest.pt", epoch, ckpt_metrics)
                    self._log_csv(epoch, "train", train_metrics)
                    if val_metrics is not None:
                        val_rmse = self._combined_rmse(val_metrics)
                        if val_rmse == val_rmse and val_rmse < self.best_val_rmse:
                            self.best_val_rmse = val_rmse
                            self._best_epoch_for_es = epoch
                            self._save_checkpoint("best.pt", epoch, val_metrics)
                            print(f"  [new best] combined_val_rmse={val_rmse:.4f}")

                        self._log_csv(epoch, "val", val_metrics)

                        if self.early_stopping_patience is not None and self._best_epoch_for_es is not None:
                            stalled = epoch - self._best_epoch_for_es
                            if stalled >= self.early_stopping_patience:
                                print(
                                    f"  [early stopping] no val improvement for {stalled} epochs "
                                    f"(patience={self.early_stopping_patience}); stopping at epoch {epoch}."
                                )
                                should_stop = True
                    self._csv_file.flush()

                stop_flag = torch.zeros(1, dtype=torch.int32, device=self.device)
                if self.is_main and should_stop:
                    stop_flag[0] = 1
                if self.ddp:
                    dist.broadcast(stop_flag, src=0)
                if stop_flag.item():
                    break

            # ── post-training test evaluation (writer/CSV still open) ─────────
            if self.is_main and test_loader is not None:
                self._run_test_eval(test_loader)

        finally:
            if self.is_main and self._csv_file:
                self._csv_file.close()
            if self.is_main and self.writer:
                self.writer.close()

        if self.is_main:
            elapsed = time.time() - t0
            print(f"\nTraining finished in {elapsed/60:.1f} min. Best val RMSE: {self.best_val_rmse:.4f}")

    # ── epoch loop ────────────────────────────────────────────────────────────

    def _run_epoch(
        self, loader: DataLoader, epoch: int, training: bool
    ) -> Dict[str, float]:
        phase = "train" if training else "val"
        self.model.train(training)

        rm = self._train_rm if training else self._val_rm
        rm.reset()

        total_loss = 0.0
        total_support = 0.0
        n_batches = 0

        # Evidential-only accumulators (val phase)
        _ev_nll_sum   = 0.0
        _ev_lr_sum    = 0.0
        _ev_sigma_sum = 0.0
        _ev_evid_sum  = 0.0
        _ev_n_valid   = 0
        # For Pearson(σ, |err|): accumulate pixelwise products
        _ev_s_sum   = 0.0
        _ev_e_sum   = 0.0
        _ev_s2_sum  = 0.0
        _ev_e2_sum  = 0.0
        _ev_se_sum  = 0.0

        self.optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            pbar = tqdm(
                loader,
                desc=f"Ep {epoch:3d} [{phase:5s}]",
                leave=False,
                disable=not self.is_main,
            )
            for batch_idx, (x, y, mask) in enumerate(pbar):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)

                if self.amp:
                    with torch.cuda.amp.autocast():
                        pred = self.model(x)
                        loss = self.criterion(pred, y, mask) / self.grad_accum
                else:
                    pred = self.model(x)
                    loss = self.criterion(pred, y, mask) / self.grad_accum

                if training:
                    if self.amp:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if (batch_idx + 1) % self.grad_accum == 0:
                        if self.amp:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            # GradScaler may skip optimizer.step() on inf/NaN grads; only
                            # then call lr_scheduler.step() after a real step (PyTorch warning).
                            scale_before = self.scaler.get_scale()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            optimizer_stepped = self.scaler.get_scale() >= scale_before
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.optimizer.step()
                            optimizer_stepped = True
                        # PyTorch requires: optimizer.step() → scheduler.step() → zero_grad()
                        if self._scheduler is not None and optimizer_stepped:
                            self._scheduler.step()
                        self.optimizer.zero_grad()
                        self.global_step += 1

                        if self.is_main and self.global_step % self.log_every == 0:
                            self.writer.add_scalar(
                                f"{phase}/step_loss",
                                loss.item() * self.grad_accum,
                                self.global_step,
                            )

                total_loss += loss.item() * self.grad_accum
                total_support += mask.float().mean().item()
                n_batches += 1

                # streaming metrics update (CPU, no tensor stacking)
                rm.update(self._mean_pred(pred).detach().cpu(), y.detach().cpu(), mask.detach().cpu())

                # ── evidential extras (val only, no grad needed) ──────────────
                if self._is_evidential and not training:
                    with torch.no_grad():
                        _p = pred.detach().float()
                        _y = y.detach().float()
                        _m = mask.detach()
                        _n_t = _y.shape[1]
                        for _ti in range(_n_t):
                            _g  = _p[:, 4 * _ti + 0][_m]
                            _nu = _p[:, 4 * _ti + 1][_m]
                            _al = _p[:, 4 * _ti + 2][_m]
                            _be = _p[:, 4 * _ti + 3][_m]
                            _yt = _y[:, _ti][_m]
                            _n  = int(_m.sum())
                            if _n == 0:
                                continue
                            import math as _math
                            _om = 2.0 * _be * (1.0 + _nu)
                            _nll_px = (
                                0.5 * torch.log(_math.pi / _nu)
                                - _al * torch.log(_om)
                                + (_al + 0.5) * torch.log((_yt - _g) ** 2 * _nu + _om)
                                + torch.lgamma(_al) - torch.lgamma(_al + 0.5)
                            )
                            _ev_nll_sum += float(_nll_px.sum())
                            _ev_lr_sum  += float((torch.abs(_yt - _g) * (2.0 * _nu + _al)).sum())
                            _sig = torch.sqrt(
                                _be / (_al - 1.0) + _be / (_nu * (_al - 1.0))
                            ).clamp(min=0.0)
                            _ae  = torch.abs(_g - _yt)
                            _ev_sigma_sum += float(_sig.sum())
                            _ev_evid_sum  += float((2.0 * _nu + _al).sum())
                            _ev_n_valid   += _n
                            # running Pearson products
                            _s = _sig.float();  _e = _ae.float()
                            _ev_s_sum  += float(_s.sum());  _ev_e_sum  += float(_e.sum())
                            _ev_s2_sum += float((_s * _s).sum())
                            _ev_e2_sum += float((_e * _e).sum())
                            _ev_se_sum += float((_s * _e).sum())

                if self.is_main:
                    pbar.set_postfix({"loss": f"{loss.item() * self.grad_accum:.4f}"})

        avg_loss = total_loss / max(n_batches, 1)
        avg_support = total_support / max(n_batches, 1)
        metrics = rm.compute()
        metrics["loss"] = avg_loss
        metrics["support_ratio"] = avg_support

        # ── attach evidential scalars to val metrics ───────────────────────
        if self._is_evidential and not training and _ev_n_valid > 0:
            import math as _math
            metrics["nll"]          = _ev_nll_sum  / _ev_n_valid
            metrics["loss_reg"]     = _ev_lr_sum   / _ev_n_valid
            metrics["mean_sigma"]   = _ev_sigma_sum / _ev_n_valid
            metrics["mean_evidence"]= _ev_evid_sum / _ev_n_valid
            # Pearson(σ, |err|)
            _N = _ev_n_valid
            _cov = _ev_se_sum / _N - (_ev_s_sum / _N) * (_ev_e_sum / _N)
            _ss  = _math.sqrt(max(_ev_s2_sum / _N - (_ev_s_sum / _N) ** 2, 0.0))
            _se  = _math.sqrt(max(_ev_e2_sum / _N - (_ev_e_sum / _N) ** 2, 0.0))
            metrics["pearson_sigma_abs_err"] = _cov / (_ss * _se + 1e-12)

        if self.is_main:
            self.writer.add_scalar(f"{phase}/loss", avg_loss, epoch)
            self.writer.add_scalar(f"{phase}/support_ratio", avg_support, epoch)
            self.writer.add_scalar(f"{phase}/lr", self.optimizer.param_groups[0]["lr"], epoch)
            for k, v in metrics.items():
                if k not in ("loss", "support_ratio") and v == v:
                    self.writer.add_scalar(f"{phase}/{k}", v, epoch)

            target_parts = "  ".join(
                f"rmse_{t[:2]}={metrics.get(f'rmse_{t}', float('nan')):.4f} "
                f"r2_{t[:2]}={metrics.get(f'r2_{t}', float('nan')):.3f}"
                for t in self._target_names
            )
            print(
                f"  Ep {epoch:3d} [{phase:5s}]  "
                f"loss={avg_loss:.4f}  "
                f"{target_parts}  "
                f"sup={avg_support:.3f}"
            )
        return metrics

    # ── test evaluation ───────────────────────────────────────────────────────

    def evaluate_split(
        self,
        loader: DataLoader,
        model: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """Inference-only pass over `loader`. Returns full metrics dict (no loss)."""
        if model is None:
            model = self.model.module if isinstance(self.model, DDP) else self.model

        inv_transforms = get_inverse_transforms(self.cfg, self._target_names)
        rm = RunningMetrics(inv_transforms, target_names=self._target_names)
        total_support = 0.0
        n_batches = 0

        model.eval()
        with torch.no_grad():
            for x, y, mask in tqdm(loader, desc="Eval [test]", leave=False, disable=not self.is_main):
                x = x.to(self.device, non_blocking=True)
                # Full-precision forward: AMP fp16 logits + expm1() blow up orig-scale RMSE.
                if self.device.type == "cuda":
                    with torch.cuda.amp.autocast(enabled=False):
                        pred = model(x)
                else:
                    pred = model(x)
                pred = pred.float()
                total_support += mask.float().mean().item()
                n_batches += 1
                rm.update(self._mean_pred(pred).detach().cpu(), y.detach().cpu(), mask.detach().cpu())

        metrics = rm.compute()
        metrics["support_ratio"] = total_support / max(n_batches, 1)
        return metrics

    def _run_test_eval(self, test_loader: DataLoader) -> None:
        """Load best.pt, evaluate on test split, log and save results. Main process only."""
        best_ckpt_path = self.save_dir / "best.pt"
        if not best_ckpt_path.exists():
            print("  [test] best.pt not found – skipping test evaluation.")
            return

        ckpt = torch.load(best_ckpt_path, map_location=self.device)
        best_epoch = int(ckpt.get("epoch", -1))

        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        raw_model.load_state_dict(ckpt["model_state_dict"])

        print(f"\nRunning test evaluation with best checkpoint (epoch {best_epoch})...")
        test_metrics = self.evaluate_split(test_loader, raw_model)

        # TensorBoard – log at the best epoch step so it aligns with train/val curves
        for k, v in test_metrics.items():
            if k != "support_ratio" and v == v:  # skip NaN
                self.writer.add_scalar(f"test/{k}", v, best_epoch)

        # CSV
        self._log_csv(best_epoch, "test", test_metrics)
        self._csv_file.flush()

        # Formatted console table
        self._print_test_table(best_epoch, test_metrics)

        # JSON alongside the best checkpoint – include provenance metadata
        out_path = self.save_dir / "test_metrics.json"
        payload = {
            "epoch": best_epoch,
            "checkpoint": str(best_ckpt_path),
            "n_patches": len(test_loader.dataset),
        }
        payload.update({k: (v if v == v else None) for k, v in test_metrics.items()})
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  Test metrics saved → {out_path}")

    def _print_test_table(self, epoch: int, metrics: Dict[str, float]) -> None:
        W = 66
        print(f"\n{'─' * W}")
        print(f"  Test evaluation  (best checkpoint: epoch {epoch})")
        print(f"{'─' * W}")
        print(f"  Support ratio  : {metrics.get('support_ratio', float('nan')):.4f}")
        print(f"  Note: *_orig uses clamped expm1 (no overflow from regression tails).")
        print(f"{'─' * W}")

        _unit_map = {"tree_count": "trees", "mean_height": "m"}
        _label_map = {"tree_count": "log1p-count", "mean_height": "log1p-height"}

        for tgt in self._target_names:
            label_log = _label_map.get(tgt, f"log1p-{tgt}")
            unit = _unit_map.get(tgt, "")

            rmse = metrics.get(f"rmse_{tgt}", float("nan"))
            mae  = metrics.get(f"mae_{tgt}",  float("nan"))
            r2   = metrics.get(f"r2_{tgt}",   float("nan"))

            r2_str = f"{r2:.4f}"
            # R² near or below 0 on tree_count is expected: on valid (tree) pixels
            # tree_count is 1–8 with median=1, giving tiny SS_tot ≈ RMSE² → R² ≈ 0.
            if tgt == "tree_count" and r2 < 0.05:
                r2_str += "  ← low (see note)"

            print(f"\n  {label_log} space:")
            print(f"    RMSE = {rmse:.4f}   MAE = {mae:.4f}   R² = {r2_str}")

            rmse_o = metrics.get(f"rmse_{tgt}_orig")
            if rmse_o is not None:
                mae_o = metrics.get(f"mae_{tgt}_orig", float("nan"))
                r2_o  = metrics.get(f"r2_{tgt}_orig",  float("nan"))
                print(f"  original scale ({unit}):")
                print(f"    RMSE = {rmse_o:.4f}   MAE = {mae_o:.4f}   R² = {r2_o:.4f}")

        if "tree_count" in self._target_names:
            print(f"\n{'─' * W}")
            print("  Why R²(tree_count) is near/below zero:")
            print("    On valid (tree) pixels, tree_count = 1–8, median = 1, ~50% of")
            print("    pixels equal 1. After log1p the target std ≈ 0.26 — nearly the")
            print("    same magnitude as RMSE. SS_tot is therefore tiny, so R² ≈ 0 even")
            print("    when RMSE is reasonable. Use RMSE / MAE as the primary metric for")
            print("    tree_count; R² is unreliable for this near-constant distribution.")
        print(f"{'─' * W}\n")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _mean_pred(self, pred: torch.Tensor) -> torch.Tensor:
        """Return the point-prediction tensor regardless of head type.

        For evidential heads the output is [B, N_t*4, H, W] where channel 4i
        is γᵢ (the predicted mean). This extracts every 4th channel so metrics
        are always computed on [B, N_t, H, W] regardless of head type.
        """
        return pred[:, 0::4] if self._is_evidential else pred

    def _build_scheduler(self, train_loader: DataLoader) -> None:
        name = self._scheduler_name
        total_steps = self.epochs * len(train_loader) // self.grad_accum
        lr = float(self.cfg.get("training", {}).get("lr", 1e-4))

        if name == "cosine":
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=lr * 0.01
            )
        elif name == "onecycle":
            self._scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=lr * 10, total_steps=total_steps
            )
        else:
            self._scheduler = None

    def _combined_rmse(self, metrics: Dict[str, float]) -> float:
        """Average RMSE over all active targets. Works for 1 or 2 targets."""
        values = []
        for t in self._target_names:
            v = metrics.get(f"rmse_{t}", float("inf"))
            if v != v:   # NaN check
                return float("inf")
            values.append(v)
        if not values:
            return float("inf")
        return sum(values) / len(values)

    def _save_checkpoint(self, filename: str, epoch: int, metrics: Dict) -> None:
        path = self.save_dir / filename
        # Save the unwrapped model state so checkpoints are DDP-independent
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "cfg": self.cfg,
            },
            path,
        )

    def _open_csv(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(self.csv_path, "w", newline="")
        fieldnames = ["epoch", "phase", "loss"]
        for t in self._target_names:
            fieldnames += [f"rmse_{t}", f"mae_{t}", f"r2_{t}"]
        for t in self._target_names:
            fieldnames += [f"rmse_{t}_orig", f"mae_{t}_orig", f"r2_{t}_orig"]
        fieldnames += ["support_ratio", "lr"]
        if self._is_evidential:
            fieldnames += ["nll", "loss_reg", "mean_sigma", "mean_evidence",
                           "pearson_sigma_abs_err"]
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=fieldnames, extrasaction="ignore"
        )
        self._csv_writer.writeheader()

    def _log_csv(self, epoch: int, phase: str, metrics: Dict) -> None:
        row = {"epoch": epoch, "phase": phase, "lr": self.optimizer.param_groups[0]["lr"]}
        row.update(metrics)
        self._csv_writer.writerow(row)

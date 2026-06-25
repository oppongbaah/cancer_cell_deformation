"""
Training, validation, and evaluation loop for the U-Net segmentation model.

Loss function: combined Binary Cross-Entropy + Dice loss, which penalises both
pixel-level and region-level errors.  This combination is standard for
biomedical image segmentation with small, imbalanced foreground regions.

Evaluation metrics (all computed on the validation set after each epoch):
  - Dice Similarity Coefficient (DSC)
  - Intersection-over-Union (IoU / Jaccard index)
  - Pixel-level Precision, Recall, Accuracy

Checkpointing: the best-validation-DSC model is saved automatically.
A snapshot of the active config is written to the checkpoint directory at
the start of training for full experiment reproducibility.

HPC note: pass ``device="cuda"`` in the config and set num_workers > 0
to fully utilise GPU + multi-core prefetching on Anvil / HPC clusters.
"""

from __future__ import annotations

import dataclasses
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from celldform.config import CelldformConfig, TrainingCfg
from celldform.segmentation.unet import UNet
from celldform.utils.metrics import SegmentationMetrics


def _print_platform_info(device: str, label: str) -> None:
    """Print OS, Python, PyTorch, and compute-device info."""
    print(f"\n{'='*60}")
    print(f"  Platform info [{label}]")
    print(f"{'='*60}")
    print(f"  OS          : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  PyTorch     : {torch.__version__}")
    if device.startswith("cuda") and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        mem_gb = props.total_memory / 1024**3
        print(f"  Device      : {props.name} (GPU {idx})")
        print(f"  CUDA        : {torch.version.cuda}")
        print(f"  VRAM        : {mem_gb:.1f} GB")
    else:
        print(f"  Device      : CPU ({platform.processor() or 'unknown'})")
    print(f"{'='*60}\n")


class _DiceBCELoss(nn.Module):
    """Weighted sum of Dice loss and Binary Cross-Entropy loss.

    alpha controls the trade-off: loss = alpha * BCE + (1 - alpha) * Dice.
    """

    def __init__(self, alpha: float = 0.3, smooth: float = 1.0, pos_weight: float = 100.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        p = probs.view(probs.size(0), -1)
        t = targets.view(targets.size(0), -1)
        intersection = (p * t).sum(dim=1)
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            p.sum(dim=1) + t.sum(dim=1) + self.smooth
        )

        return self.alpha * bce_loss + (1.0 - self.alpha) * dice_loss.mean()


class SegmentationTrainer:
    """Orchestrates the U-Net training loop.

    Accepts a :class:`~celldform.config.CelldformConfig` object so that
    all training hyper-parameters, optimizer choice, scheduler, checkpoint
    behaviour, and resume logic are driven entirely from the config file.

    Parameters
    ----------
    model:
        A :class:`UNet` instance.
    train_loader:
        DataLoader yielding ``(image, mask)`` batches for training.
    val_loader:
        DataLoader yielding ``(image, mask)`` batches for validation.
    conf:
        Fully populated :class:`CelldformConfig` (from :func:`celldform.config.load`).
    """

    def __init__(
        self,
        model: UNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        conf: CelldformConfig,
    ) -> None:
        self.conf = conf
        self.device = conf.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(conf.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        on_cuda = self.device.startswith("cuda")
        self.use_amp = conf.training.use_amp and on_cuda
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        if on_cuda:
            torch.backends.cudnn.benchmark = True  # fixed 256×256 input → cuDNN picks optimal kernel once

        self.criterion = _DiceBCELoss(
            alpha=conf.training.loss_alpha,
            pos_weight=conf.training.loss_pos_weight,
        ).to(self.device)
        self.optimizer = _build_optimizer(model, conf.training)
        self.scheduler = _build_scheduler(self.optimizer, conf.training)
        self.metrics = SegmentationMetrics()

        # Resume from checkpoint if configured.
        self.start_epoch = 1
        if conf.resume.checkpoint:
            self.start_epoch = self._load_resume()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, epochs: Optional[int] = None) -> Dict[str, list]:
        """Run the full training loop.

        Parameters
        ----------
        epochs:
            Number of epochs.  Defaults to ``conf.training.epochs`` when
            *None*, so callers only need to pass this to override the config.

        Returns
        -------
        History dict with keys ``train_loss``, ``val_loss``, ``val_dsc``.
        """
        n_epochs = epochs if epochs is not None else self.conf.training.epochs
        history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_dsc": []}
        best_dsc = 0.0

        # Snapshot the config so the checkpoint folder is self-contained.
        _save_config_snapshot(self.conf, self.checkpoint_dir)

        _print_platform_info(self.device, "before training")

        for epoch in range(self.start_epoch, n_epochs + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch()
            val_loss, val_dsc = self._validate()
            self._step_scheduler(val_dsc)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_dsc"].append(val_dsc)

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:03d}/{n_epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_DSC={val_dsc:.4f}  ({elapsed:.1f}s)"
            )

            if val_dsc > best_dsc:
                best_dsc = val_dsc
                self._save_checkpoint("unet_best.pt", epoch, val_dsc)

        _print_platform_info(self.device, "after training")

        return history

    def evaluate(self, loader=None) -> Dict[str, float]:
        """Compute all segmentation metrics on a DataLoader.

        Parameters
        ----------
        loader:
            Any DataLoader of (image, mask) pairs.  Defaults to the
            validation loader used during training when *None*.
        """
        loader = loader or self.val_loader
        self.model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_amp):
            for images, masks in loader:
                images = images.to(self.device, non_blocking=True)
                logits = self.model(images)
                preds = (torch.sigmoid(logits) >= 0.5).float().cpu()
                all_preds.append(preds)
                all_targets.append(masks.cpu())

        return self.metrics.compute_all(
            torch.cat(all_preds), torch.cat(all_targets)
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for images, masks in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).float()
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                loss = self.criterion(self.model(images), masks)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item() * images.size(0)

        return total_loss / len(self.train_loader.dataset)

    def _validate(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_dsc = 0.0

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_amp):
            for images, masks in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True).float()
                logits = self.model(images)
                total_loss += self.criterion(logits, masks).item() * images.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                total_dsc += self.metrics.dice(preds.cpu(), masks.cpu()) * images.size(0)

        n = len(self.val_loader.dataset)
        return total_loss / n, total_dsc / n

    def _step_scheduler(self, val_dsc: float) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_dsc)
        else:
            self.scheduler.step()

    def _save_checkpoint(self, filename: str, epoch: int, val_dsc: float) -> None:
        path = self.checkpoint_dir / filename
        ckpt: Dict = {
            "epoch": epoch,
            "val_dsc": val_dsc,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(ckpt, path)
        print(f"  → Checkpoint saved: {path}  (DSC={val_dsc:.4f})")

    def _load_resume(self) -> int:
        """Load weights/state from resume checkpoint, return next epoch number."""
        path = Path(self.conf.resume.checkpoint)
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])

        if self.conf.resume.load_optimizer and "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if (
            self.conf.resume.load_scheduler
            and self.scheduler is not None
            and "scheduler_state_dict" in ckpt
        ):
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        next_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[celldform] Resumed from {path}  (next epoch: {next_epoch})")
        return next_epoch


# ---------------------------------------------------------------------------
# Optimizer / scheduler factories
# ---------------------------------------------------------------------------

def _build_optimizer(
    model: nn.Module, cfg: TrainingCfg
) -> torch.optim.Optimizer:
    name = cfg.optimizer
    params = model.parameters()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9)
    raise ValueError(f"Unknown optimizer '{name}'. Choose from: adamw, adam, sgd.")


def _build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: TrainingCfg
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    name = cfg.scheduler
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max",
            patience=cfg.scheduler_patience,
            factor=cfg.scheduler_factor,
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01,
        )
    if name == "none":
        return None
    raise ValueError(
        f"Unknown scheduler '{name}'. Choose from: reduce_on_plateau, cosine, none."
    )


# ---------------------------------------------------------------------------
# Config snapshot
# ---------------------------------------------------------------------------

def _save_config_snapshot(conf: CelldformConfig, checkpoint_dir: Path) -> None:
    """Write the active config to the checkpoint directory for reproducibility."""
    snapshot = dataclasses.asdict(conf)
    path = checkpoint_dir / "config.yaml"
    with open(path, "w") as fh:
        yaml.dump(snapshot, fh, default_flow_style=False, sort_keys=False)
    print(f"[celldform] Config snapshot → {path}")

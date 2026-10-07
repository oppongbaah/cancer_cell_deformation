"""
Training, validation, and evaluation loop for the U-Net segmentation model.

Loss function: combined Cross-Entropy + region loss (Dice, or Focal-Tversky
via training.loss_type), which penalises both
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
to fully utilise GPU + multi-core prefetching on the NSU cluster / HPC.
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
import torch.nn.functional as F
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


class _DiceCELoss(nn.Module):
    """Weighted sum of multiclass Dice loss and Cross-Entropy loss.

    Multiclass counterpart to :class:`_DiceBCELoss`: loss = alpha * CE +
    (1 - alpha) * Dice. The Dice term averages over all classes including
    background by default; pass exclude_background=True to average over the
    foreground classes only.
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.3,
        smooth: float = 1.0,
        class_weights: Optional[list] = None,
        exclude_background: bool = False,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.num_classes = num_classes
        self.exclude_background = exclude_background
        weight = torch.tensor(class_weights) if class_weights is not None else None
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        # logits: (B, C, H, W) raw class logits; targets: (B, H, W) integer class ids
        ce_loss = self.ce(logits, targets)

        probs = torch.softmax(logits, dim=1)
        target_onehot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        p = probs.reshape(probs.size(0), self.num_classes, -1)
        t = target_onehot.reshape(target_onehot.size(0), self.num_classes, -1)
        intersection = (p * t).sum(dim=2)
        dice_per_class = (2.0 * intersection + self.smooth) / (
            p.sum(dim=2) + t.sum(dim=2) + self.smooth
        )
        # Background (class 0) is ~99.96% correct from pixel imbalance alone and barely
        # moves, so including it in this mean spends ~1/3 of the Dice term on a constant.
        if self.exclude_background:
            dice_per_class = dice_per_class[:, 1:]
        dice_loss = 1.0 - dice_per_class.mean()

        return self.alpha * ce_loss + (1.0 - self.alpha) * dice_loss


class _FocalTverskyCELoss(nn.Module):
    """Weighted sum of Cross-Entropy and a Focal-Tversky region term.

    Same structure as :class:`_DiceCELoss` -- ``loss = alpha * CE + (1 - alpha) * region``
    -- with the Dice region term replaced by Focal-Tversky::

        TI_c = (TP_c + s) / (TP_c + tversky_alpha * FP_c + tversky_beta * FN_c + s)
        FTL  = mean_c (1 - TI_c) ** focal_gamma

    The Tversky index generalises Dice, which is the special case
    ``tversky_alpha == tversky_beta == 0.5``. Setting ``tversky_alpha > tversky_beta``
    makes a false positive cost more than a false negative, directly targeting
    over-segmentation. ``focal_gamma == 1.0`` reduces this to plain Tversky.

    Note ``class_weights`` applies to the CE term only, exactly as in _DiceCELoss --
    the region term is weighted through tversky_alpha/beta instead.
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.3,
        smooth: float = 1.0,
        class_weights: Optional[list] = None,
        exclude_background: bool = False,
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3,
        focal_gamma: float = 0.75,
        lovasz_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.lovasz_weight = lovasz_weight
        self.alpha = alpha
        self.smooth = smooth
        self.num_classes = num_classes
        self.exclude_background = exclude_background
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.focal_gamma = focal_gamma
        weight = torch.tensor(class_weights) if class_weights is not None else None
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)

        probs = torch.softmax(logits, dim=1)
        target_onehot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        p = probs.reshape(probs.size(0), self.num_classes, -1)
        t = target_onehot.reshape(target_onehot.size(0), self.num_classes, -1)

        tp = (p * t).sum(dim=2)
        fp = (p * (1.0 - t)).sum(dim=2)
        fn = ((1.0 - p) * t).sum(dim=2)

        ti = (tp + self.smooth) / (
            tp + self.tversky_alpha * fp + self.tversky_beta * fn + self.smooth
        )
        if self.exclude_background:
            ti = ti[:, 1:]
        # clamp keeps the fractional power away from a zero base, whose gradient is
        # unbounded and produces NaNs under AMP once a class is segmented near-perfectly.
        ftl = ((1.0 - ti).clamp(min=1e-6) ** self.focal_gamma).mean()

        loss = self.alpha * ce_loss + (1.0 - self.alpha) * ftl
        if self.lovasz_weight > 0:
            loss = loss + self.lovasz_weight * _lovasz_softmax(probs.float(), targets)
        return loss


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovasz extension of the Jaccard loss w.r.t. sorted errors."""
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1.0 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def _lovasz_softmax(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Lovasz-softmax loss (Berman, Rannen Triki & Blaschko, CVPR 2018).

    A convex surrogate of 1 - IoU, optimised directly. Computed PER IMAGE and averaged,
    over the classes present in that image's ground truth -- so an image whose trapped
    cell is missed entirely contributes a loss near 1 for that class instead of being
    diluted across the batch. ``probs`` (B, C, H, W) softmax, ``targets`` (B, H, W).
    """
    losses = []
    for p_img, t_img in zip(probs, targets):
        p_flat = p_img.reshape(p_img.size(0), -1)             # (C, P)
        t_flat = t_img.reshape(-1)
        per_class = []
        for c in range(p_flat.size(0)):
            fg = (t_flat == c).float()
            if fg.sum() == 0:
                continue
            errors = (fg - p_flat[c]).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            per_class.append(torch.dot(errors_sorted, _lovasz_grad(fg[perm])))
        if per_class:
            losses.append(torch.stack(per_class).mean())
    return torch.stack(losses).mean() if losses else probs.sum() * 0.0


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
        val_loader: Optional[DataLoader],
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
        self.device_type = "cuda" if on_cuda else "cpu"  # autocast/GradScaler want a bare type, not e.g. "cuda:0"
        self.use_amp = conf.training.use_amp and on_cuda
        self.scaler = torch.amp.GradScaler(self.device_type, enabled=self.use_amp)
        if on_cuda:
            # cudnn.benchmark=True auto-tunes convolution kernels by timing them
            # against runtime hardware state — faster, but a source of run-to-run
            # non-determinism that `torch.manual_seed` alone does not suppress.
            # See training.cudnn_deterministic / cudnn_benchmark in config.py.
            torch.backends.cudnn.deterministic = conf.training.cudnn_deterministic
            torch.backends.cudnn.benchmark = conf.training.cudnn_benchmark
        if conf.training.strict_deterministic:
            # warn_only=True: enforce deterministic ops where a deterministic
            # implementation exists, but don't hard-crash training on the rare op
            # (e.g. certain interpolate/upsample backward paths) that has none.
            # Measured 6-7x slower per epoch on this architecture — see
            # training.strict_deterministic docstring in config.py.
            torch.use_deterministic_algorithms(True, warn_only=True)

        self.out_channels = conf.unet.out_channels
        # If binay segmentation, use Dice + BCE; if multiclass, use Dice + CE or Focal-Tversky + CE.
        if self.out_channels == 1:
            self.criterion = _DiceBCELoss(
                alpha=conf.training.loss_alpha,
                pos_weight=conf.training.loss_pos_weight,
            ).to(self.device)
        elif conf.training.loss_type == "focal_tversky":
            self.criterion = _FocalTverskyCELoss(
                num_classes=self.out_channels,
                alpha=conf.training.loss_alpha,
                class_weights=conf.training.class_weights,
                exclude_background=conf.training.dice_exclude_background,
                tversky_alpha=conf.training.tversky_alpha,
                tversky_beta=conf.training.tversky_beta,
                focal_gamma=conf.training.focal_gamma,
                lovasz_weight=conf.training.lovasz_weight,
            ).to(self.device)
        else:
            self.criterion = _DiceCELoss(
                num_classes=self.out_channels,
                alpha=conf.training.loss_alpha,
                class_weights=conf.training.class_weights,
                exclude_background=conf.training.dice_exclude_background,
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

                History dict with keys ``train_loss``, ``train_dsc``, ``val_loss``,
        ``val_dsc``.
        """
        n_epochs = epochs if epochs is not None else self.conf.training.epochs
        history: Dict[str, list] = {
            "train_loss": [], "train_dsc": [], "val_loss": [], "val_dsc": [],
        }
        best_dsc = 0.0
        best_epoch = 0
        patience = self.conf.training.early_stopping_patience
        epochs_since_improvement = 0
        best_patience_dsc = 0.0  # separate from best_dsc: requires min_delta improvement to reset the
                                  # patience counter, so noise-level upticks don't stall early stopping
                                  # forever, while unet_best.pt still saves on any improvement as before
        min_delta = 1e-3

        # Snapshot the config so the checkpoint folder is self-contained.
        _save_config_snapshot(self.conf, self.checkpoint_dir)

        _print_platform_info(self.device, "before training")

        for epoch in range(self.start_epoch, n_epochs + 1):
            t0 = time.time()
            train_loss, train_dsc = self._train_one_epoch()
            if self.val_loader is None:
                # Fixed-length run (e.g. the randswap deployment model): no cell is held out, so there is
                # nothing to select an epoch on -- train exactly n_epochs and keep the last weights.
                val_loss, val_dsc = float("nan"), float("nan")
            else:
                val_loss, val_dsc = self._validate()
            self._step_scheduler(val_dsc)

            history["train_loss"].append(train_loss)
            history["train_dsc"].append(train_dsc)
            history["val_loss"].append(val_loss)
            history["val_dsc"].append(val_dsc)

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:03d}/{n_epochs}  "
                f"train_loss={train_loss:.4f}  train_DSC={train_dsc:.4f}  "
                f"val_loss={val_loss:.4f}  val_DSC={val_dsc:.4f}  ({elapsed:.1f}s)"
            )

            if self.val_loader is None:
                if epoch == n_epochs:
                    self._save_checkpoint("unet_deploy.pt", epoch, val_dsc)
                continue

            if val_dsc > best_dsc:
                best_dsc = val_dsc
                best_epoch = epoch
                self._save_checkpoint("unet_best.pt", epoch, val_dsc)

            if val_dsc > best_patience_dsc + min_delta:
                best_patience_dsc = val_dsc
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if patience is not None and epochs_since_improvement >= patience:
                print(f"Early stopping: no val_DSC improvement > {min_delta} for {patience} epochs "
                      f"(best={best_dsc:.4f} at epoch {best_epoch}). Stopping at epoch {epoch}/{n_epochs}.")
                break

        # _print_platform_info(self.device, "after training")

        return history

    def evaluate(self, loader=None) -> Dict[str, float]:
        """Compute segmentation metrics on a DataLoader.

        Parameters
        ----------
        loader:
            Any DataLoader of (image, mask) pairs.  Defaults to the
            validation loader used during training when *None*.

        Returns
        -------
        Binary mode (``out_channels == 1``): flat dict (dsc, iou, ...).
        Multiclass mode (``out_channels > 1``): dict keyed by class id (plus
        ``"mean"``), each a flat metrics dict — see
        :meth:`SegmentationMetrics.per_class`. Class 1 is the trapped cell,
        directly comparable to the binary mode's dsc.
        """
        loader = loader or self.val_loader
        self.model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad(), torch.amp.autocast(self.device_type, enabled=self.use_amp):
            for images, masks in loader:
                images = images.to(self.device, non_blocking=True)
                logits = self.model(images)
                if self.out_channels == 1:
                    preds = (torch.sigmoid(logits) >= 0.5).float().cpu()
                else:
                    preds = torch.softmax(logits, dim=1).argmax(dim=1).cpu()
                all_preds.append(preds)
                all_targets.append(masks.cpu())

        preds_cat = torch.cat(all_preds)
        targets_cat = torch.cat(all_targets)
        if self.out_channels == 1:
            return self.metrics.compute_all(preds_cat, targets_cat)
        return self.metrics.per_class(preds_cat, targets_cat, self.out_channels)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_one_epoch(self) -> Tuple[float, float]:
        """Returns (train_loss, train_dsc). train_dsc is measured on each
        batch's predictions *before* that batch's weight update (same
        convention as train_loss), using the same batch-pooled,
        batch-size-weighted averaging as ``_validate`` so the two curves are
        computed the same way — see ``_validate`` docstring."""
        self.model.train()
        total_loss = 0.0
        total_dsc = 0.0

        target_dtype = torch.long if self.out_channels > 1 else torch.float32
        # Gradient accumulation: k micro-batches per optimizer step, each loss scaled by
        # 1/k, so batch_size=4 with k=4 takes the same-sized step as batch_size=16 when
        # memory forbids the larger batch (512x512, 2026-09-30). BatchNorm still sees the
        # micro-batch, which is why resolution comparisons need a matched-micro-batch control.
        k = max(1, self.conf.training.grad_accum_steps)
        n_batches = len(self.train_loader)
        self.optimizer.zero_grad()
        dw, T = self.conf.training.distill_weight, self.conf.training.distill_temperature
        n_seen = 0
        for i, batch in enumerate(self.train_loader):
            images, masks = batch[0], batch[1]
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).to(target_dtype)
            with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)
                if dw > 0 and len(batch) == 3:
                    # Knowledge distillation (Hinton et al. 2015): match the teacher's
                    # temperature-softened class probabilities, per pixel. T^2 restores the
                    # gradient scale the softening removes.
                    teacher = batch[2].to(self.device, non_blocking=True).float()
                    t_soft = torch.softmax(torch.log(teacher.clamp_min(1e-6)) / T, dim=1)
                    s_log = torch.log_softmax(logits.float() / T, dim=1)
                    kl = torch.nn.functional.kl_div(s_log, t_soft, reduction="none").sum(1).mean()
                    loss = loss + dw * T * T * kl
            self.scaler.scale(loss / k).backward()
            if (i + 1) % k == 0 or i + 1 == n_batches:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            total_loss += loss.item() * images.size(0)
            n_seen += images.size(0)

            with torch.no_grad():
                if self.out_channels == 1:
                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    total_dsc += self.metrics.dice(preds.cpu(), masks.cpu()) * images.size(0)
                else:
                    preds = torch.softmax(logits, dim=1).argmax(dim=1)
                    total_dsc += self.metrics.dice(
                        (preds.cpu() == 1), (masks.cpu() == 1)
                    ) * images.size(0)

        # Samples actually drawn, not the dataset size: a WeightedRandomSampler may draw fewer
        # (distillation keeps the CV epoch budget while the dataset grows ~5x), and dividing by
        # len(dataset) printed train_DSC ~0.18 for a model at ~0.95 (2026-10-01).
        return total_loss / n_seen, total_dsc / n_seen

    def _validate(self) -> Tuple[float, float]:
        """Returns (val_loss, val_dsc). For multiclass, val_dsc is the
        trapped-cell (class 1) DSC — comparable to the binary baseline."""
        self.model.eval()
        total_loss = 0.0
        total_dsc = 0.0
        target_dtype = torch.long if self.out_channels > 1 else torch.float32

        with torch.no_grad(), torch.amp.autocast(self.device_type, enabled=self.use_amp):
            for images, masks in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True).to(target_dtype)
                logits = self.model(images)
                total_loss += self.criterion(logits, masks).item() * images.size(0)
                if self.out_channels == 1:
                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    total_dsc += self.metrics.dice(preds.cpu(), masks.cpu()) * images.size(0)
                else:
                    preds = torch.softmax(logits, dim=1).argmax(dim=1)
                    total_dsc += self.metrics.dice(
                        (preds.cpu() == 1), (masks.cpu() == 1)
                    ) * images.size(0)

        n = len(self.val_loader.dataset)
        return total_loss / n, total_dsc / n

    def _step_scheduler(self, val_dsc: float) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_dsc)
        else:
            self.scheduler.step()

    def _save_checkpoint(self, filename: str, epoch: int, val_dsc: float, model: Optional[nn.Module] = None) -> None:
        path = self.checkpoint_dir / filename
        ckpt: Dict = {
            "epoch": epoch,
            "val_dsc": val_dsc,
            "model_state_dict": (model or self.model).state_dict(),
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
        return torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                               momentum=cfg.momentum, nesterov=cfg.nesterov)
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

def _yaml_safe(obj):
    """Recursively convert tuples to lists so the result round-trips through
    yaml.safe_load (used by celldform.config.load and most consumers of this
    snapshot). dataclasses.asdict() preserves tuple-typed fields (e.g.
    preprocessing.target_size) as tuples, and plain yaml.dump serializes
    those with a !!python/tuple tag that yaml.safe_load cannot parse —
    silently making the "written for reproducibility" snapshot unloadable
    by anything using the standard config loader. Found 2026-09-04 when 17
    existing checkpoint dirs' snapshots all had this problem."""
    if isinstance(obj, tuple):
        return [_yaml_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_yaml_safe(v) for v in obj]
    return obj


def _save_config_snapshot(conf: CelldformConfig, checkpoint_dir: Path) -> None:
    """Write the active config to the checkpoint directory for reproducibility."""
    snapshot = _yaml_safe(dataclasses.asdict(conf))
    path = checkpoint_dir / "config.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(snapshot, fh, default_flow_style=False, sort_keys=False)
    print(f"[celldform] Config snapshot → {path}")

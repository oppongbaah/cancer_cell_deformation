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

HPC note: pass ``device="cuda"`` and use a DataLoader with num_workers > 0
to fully utilise GPU + multi-core prefetching on Anvil / HPC clusters.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from celldform.segmentation.unet import UNet
from celldform.utils.metrics import SegmentationMetrics


class _DiceBCELoss(nn.Module):
    """Weighted sum of Dice loss and Binary Cross-Entropy loss.

    alpha controls the trade-off: loss = alpha * BCE + (1 - alpha) * Dice.
    """

    def __init__(self, alpha: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        # Flatten spatial dims for dice computation.
        p = probs.view(probs.size(0), -1)
        t = targets.view(targets.size(0), -1)
        intersection = (p * t).sum(dim=1)
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            p.sum(dim=1) + t.sum(dim=1) + self.smooth
        )
        dice_loss = dice_loss.mean()

        return self.alpha * bce_loss + (1.0 - self.alpha) * dice_loss


class SegmentationTrainer:
    """Orchestrates the U-Net training loop.

    Parameters
    ----------
    model:
        A :class:`UNet` instance (or any compatible segmentation model).
    train_loader:
        DataLoader yielding ``(image, mask)`` batches for training.
    val_loader:
        DataLoader yielding ``(image, mask)`` batches for validation.
    lr:
        Initial Adam learning rate.
    checkpoint_dir:
        Directory for saving model checkpoints.
    device:
        ``"cuda"`` or ``"cpu"``.  Auto-detected if *None*.
    """

    def __init__(
        self,
        model: UNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-4,
        checkpoint_dir: str | Path = "checkpoints",
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = _DiceBCELoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        # Halve LR when validation DSC plateaus for 5 consecutive epochs.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", patience=5, factor=0.5
        )
        self.metrics = SegmentationMetrics()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, epochs: int = 50) -> Dict[str, list]:
        """Run the full training loop for *epochs* epochs.

        Returns
        -------
        History dict with keys ``train_loss``, ``val_loss``, ``val_dsc``.
        """
        history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_dsc": []}
        best_dsc = 0.0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch()
            val_loss, val_dsc = self._validate()
            self.scheduler.step(val_dsc)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_dsc"].append(val_dsc)

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:03d}/{epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_DSC={val_dsc:.4f}  ({elapsed:.1f}s)"
            )

            if val_dsc > best_dsc:
                best_dsc = val_dsc
                self._save_checkpoint("unet_best.pt", epoch, val_dsc)

        return history

    def evaluate(self) -> Dict[str, float]:
        """Compute all segmentation metrics on the validation set."""
        self.model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                logits = self.model(images)
                preds = (torch.sigmoid(logits) >= 0.5).float().cpu()
                all_preds.append(preds)
                all_targets.append(masks.cpu())

        preds_cat = torch.cat(all_preds)
        targets_cat = torch.cat(all_targets)
        return self.metrics.compute_all(preds_cat, targets_cat)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for images, masks in self.train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device).float()

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)

        return total_loss / len(self.train_loader.dataset)

    def _validate(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_dsc = 0.0

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device).float()

                logits = self.model(images)
                loss = self.criterion(logits, masks)
                total_loss += loss.item() * images.size(0)

                preds = (torch.sigmoid(logits) >= 0.5).float()
                dsc = self.metrics.dice(preds.cpu(), masks.cpu())
                total_dsc += dsc * images.size(0)

        n = len(self.val_loader.dataset)
        return total_loss / n, total_dsc / n

    def _save_checkpoint(self, filename: str, epoch: int, val_dsc: float) -> None:
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "val_dsc": val_dsc,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        print(f"  → Checkpoint saved: {path}  (DSC={val_dsc:.4f})")

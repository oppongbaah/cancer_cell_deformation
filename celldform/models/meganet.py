"""
MegaNet — custom CNN for regression of morphological deformation parameters.

Architecture (Proposal §3.4)
----------------------------

  Input: (B, 1, H, W) single-channel microscopy frame (256×256 by default)

  Block 1:  Conv2d(1→16, 3×3, pad=1) → ReLU → MaxPool2d(2)   → (B, 16, 128, 128)
  Block 2:  Conv2d(16→32, 3×3, pad=1) → ReLU → MaxPool2d(2)  → (B, 32,  64,  64)
  Block 3:  Conv2d(32→64, 3×3, pad=1) → ReLU → MaxPool2d(2)  → (B, 64,  32,  32)

  Global Average Pooling → (B, 64)

  FC1: Linear(64→32) → ReLU → Dropout(0.3)
  FC2: Linear(32→n_outputs)   [no activation — raw regression outputs]

n_outputs defaults to 1 for a single deformation index target.
Set n_outputs > 1 to regress multiple morphological parameters simultaneously
(e.g. aspect_ratio, deformation_index, area, perimeter → n_outputs=4).

Training
--------
  Loss:      MSELoss
  Optimiser: AdamW (weight_decay=1e-4)
  Scheduler: ReduceLROnPlateau (mode="min", patience=10, factor=0.5)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class MegaNet(nn.Module):
    """Three-block CNN with Global Average Pooling for deformation regression.

    Parameters
    ----------
    in_channels:
        Number of input image channels (1 = grayscale).
    n_outputs:
        Number of regression targets.  1 → predict deformation index only.
        >1 → multi-output (aspect_ratio, deformation_index, area, …).
    dropout_p:
        Dropout probability applied before the final linear layer.
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_outputs: int = 1,
        dropout_p: float = 0.3,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            _conv_block(in_channels, 16),   # → (B, 16, H/2,  W/2)
            _conv_block(16, 32),            # → (B, 32, H/4,  W/4)
            _conv_block(32, 64),            # → (B, 64, H/8,  W/8)
        )

        # Global Average Pooling collapses spatial dims to a 64-D descriptor.
        self.gap = nn.AdaptiveAvgPool2d(1)  # → (B, 64, 1, 1)

        self.regressor = nn.Sequential(
            nn.Flatten(),                   # → (B, 64)
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(32, n_outputs),       # raw regression output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return shape (B, n_outputs) regression predictions."""
        x = self.features(x)
        x = self.gap(x)
        return self.regressor(x)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        image: "np.ndarray",    # noqa: F821
        device: Optional[str] = None,
    ) -> "np.ndarray":
        """Predict deformation parameters for a single preprocessed image.

        Parameters
        ----------
        image:
            Float32 array of shape (H, W) or (1, H, W) in [0, 1].

        Returns
        -------
        1-D numpy array of length n_outputs.
        """
        import numpy as np

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(dev).eval()

        if image.ndim == 2:
            image = image[np.newaxis, np.newaxis, ...]
        elif image.ndim == 3:
            image = image[np.newaxis, ...]

        tensor = torch.from_numpy(image).float().to(dev)
        out = self(tensor)
        return out.squeeze().cpu().numpy()

    # ------------------------------------------------------------------
    # Serialisation (delegated to CheckpointManager but also standalone)
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        torch.save(self.state_dict(), Path(path))

    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, Path],
        in_channels: int = 1,
        n_outputs: int = 1,
        device: Optional[str] = None,
    ) -> "MegaNet":
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = cls(in_channels=in_channels, n_outputs=n_outputs)
        model.load_state_dict(torch.load(Path(path), map_location=dev))
        return model.to(dev).eval()


class MegaNetTrainer:
    """Training loop for MegaNet regression.

    Parameters
    ----------
    model:
        A :class:`MegaNet` instance.
    train_loader, val_loader:
        DataLoaders yielding ``(image, targets)`` batches.
        ``targets`` shape: (B,) for single output or (B, n_outputs) for multi.
    lr:
        Initial learning rate.
    checkpoint_dir:
        Directory for saving the best-validation-loss checkpoint.
    device:
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        model: MegaNet,
        train_loader,
        val_loader,
        lr: float = 1e-3,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=10, factor=0.5
        )

    def fit(self, epochs: int = 100) -> Dict[str, list]:
        """Run training loop, return loss/MAE history."""
        history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_mae": []}
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch()
            val_loss, val_mae = self._val_epoch()
            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_mae"].append(val_mae)

            print(
                f"Epoch {epoch:03d}/{epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_MAE={val_mae:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"epoch": epoch, "val_loss": val_loss, "model_state_dict": self.model.state_dict()},
                    self.checkpoint_dir / "meganet_best.pt",
                )

        return history

    def _train_epoch(self) -> float:
        self.model.train()
        total = 0.0
        for images, targets in self.train_loader:
            images = images.to(self.device)
            targets = targets.to(self.device).float()
            if targets.ndim == 1:
                targets = targets.unsqueeze(1)
            self.optimizer.zero_grad()
            preds = self.model(images)
            loss = self.criterion(preds, targets)
            loss.backward()
            self.optimizer.step()
            total += loss.item() * images.size(0)
        return total / len(self.train_loader.dataset)

    def _val_epoch(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss, total_mae = 0.0, 0.0
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device).float()
                if targets.ndim == 1:
                    targets = targets.unsqueeze(1)
                preds = self.model(images)
                total_loss += self.criterion(preds, targets).item() * images.size(0)
                total_mae += F.l1_loss(preds, targets).item() * images.size(0)
        n = len(self.val_loader.dataset)
        return total_loss / n, total_mae / n

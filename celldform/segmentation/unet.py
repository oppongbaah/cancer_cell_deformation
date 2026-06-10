"""
U-Net architecture for pixel-level cancer cell segmentation.

Architecture follows Ronneberger et al. (2015) with minor adaptations:
  - Input  : (B, 1, H, W) grayscale image (H = W = 256 by default)
  - Output : (B, 1, H, W) binary segmentation mask logit map
  - Encoder: 4 downsampling blocks doubling channels (64 → 128 → 256 → 512)
  - Bottleneck: 1024 channels
  - Decoder: 4 upsampling blocks with skip connections from encoder

The encoder/decoder use BatchNorm + ReLU after each 3×3 conv, which improves
convergence on small biomedical datasets compared to the original design.

Reference
---------
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks
for Biomedical Image Segmentation. MICCAI 2015, 234–241.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Module):
    """Two consecutive (Conv → BN → ReLU) layers — the core U-Net block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):
    """Encoder step: MaxPool2d(2) → DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class _Up(nn.Module):
    """Decoder step: bilinear upsample → concatenate skip → DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        # Halve the channels before concatenation so the concat gives in_ch total.
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad x to match skip's spatial size if they differ due to odd dimensions.
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """Encoder–decoder U-Net for binary cell segmentation.

    Parameters
    ----------
    in_channels:
        Number of input image channels (1 = grayscale, 3 = RGB).
    base_features:
        Channel count in the first encoder block; doubles at each depth level.
    """

    def __init__(self, in_channels: int = 1, base_features: int = 64) -> None:
        super().__init__()
        f = base_features

        # Encoder
        self.enc1 = _DoubleConv(in_channels, f)
        self.enc2 = _Down(f, f * 2)
        self.enc3 = _Down(f * 2, f * 4)
        self.enc4 = _Down(f * 4, f * 8)

        # Bottleneck
        self.bottleneck = _Down(f * 8, f * 16)

        # Decoder
        self.dec4 = _Up(f * 16, f * 8)
        self.dec3 = _Up(f * 8, f * 4)
        self.dec2 = _Up(f * 4, f * 2)
        self.dec1 = _Up(f * 2, f)

        # 1×1 conv to project to a single-channel binary mask logit
        self.out_conv = nn.Conv2d(f, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logit mask — apply sigmoid externally for probabilities."""
        # Encoder path
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        # Bottleneck
        b = self.bottleneck(s4)

        # Decoder path with skip connections
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.out_conv(d1)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        image: "np.ndarray",  # noqa: F821 — avoid circular import
        threshold: float = 0.5,
        device: Optional[str] = None,
    ) -> "np.ndarray":
        """Segment a single preprocessed NumPy image, return binary mask.

        Parameters
        ----------
        image:
            Float32 array of shape (H, W) or (1, H, W) in [0, 1].
        threshold:
            Sigmoid probability above which a pixel is classified as cell.
        device:
            ``"cuda"`` or ``"cpu"``.  Auto-detected if *None*.
        """
        import numpy as np

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(dev).eval()

        if image.ndim == 2:
            image = image[np.newaxis, np.newaxis, ...]   # (1, 1, H, W)
        elif image.ndim == 3:
            image = image[np.newaxis, ...]               # (1, C, H, W)

        tensor = torch.from_numpy(image).float().to(dev)
        logit = self(tensor)                             # (1, 1, H, W)
        prob = torch.sigmoid(logit).squeeze().cpu().numpy()
        return (prob >= threshold).astype(np.uint8)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist model weights to *path*."""
        torch.save(self.state_dict(), Path(path))

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        in_channels: int = 1,
        base_features: int = 64,
        device: Optional[str] = None,
    ) -> "UNet":
        """Load a previously saved U-Net from a checkpoint file."""
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = cls(in_channels=in_channels, base_features=base_features)
        model.load_state_dict(torch.load(Path(path), map_location=dev))
        return model.to(dev).eval()

"""Run several trained U-Nets as one, by averaging their predictions.

This is **not** a merged model. It holds N separate checkpoints and runs all N on
every frame, averaging their per-class probabilities. Nothing is combined at the
weight level -- averaging weights across independently seeded networks does not
work (measured: DSC 0.000), because they occupy different loss basins and their
hidden units are not aligned. Only the outputs live in a shared, comparable space.

Why bother: on the 2026-09-24 four-seed baseline a single model scores 0.809 +/- 0.049
evaluated, and two runs in four lose one development cell entirely (0.507 and 0.630)
on an unpredictable cell. The ensemble scores 0.859 and no per-cell collapse survives
it -- the healthy members outvote the one that failed.

Cost, measured on an RTX 4060 at 256x256: 11.9 ms/frame for one model, 46.2 ms for
four (3.89x), against a 22 fps source. So four fits real time with little headroom.

The class exposes ``predict()`` with the same signature as :class:`UNet`, so
``RealTimePipeline(unet=...)`` and ``scripts/inference/infer.py`` take it unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence

import torch

from celldform.segmentation.unet import UNet

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


class EnsembleUNet:
    """A drop-in stand-in for :class:`UNet` backed by several checkpoints.

    Parameters
    ----------
    models:
        Already-loaded :class:`UNet` instances. All must share ``out_channels``.
    device:
        ``"cuda"`` or ``"cpu"``. Auto-detected when *None*.
    """

    def __init__(self, models: Sequence[UNet], device: Optional[str] = None) -> None:
        if not models:
            raise ValueError("EnsembleUNet needs at least one model")
        # UNet does not keep out_channels as an attribute -- it only sizes the final
        # 1x1 conv with it, so read it back from there.
        out = {m.out_conv.out_channels for m in models}
        if len(out) != 1:
            raise ValueError(f"members disagree on out_channels: {sorted(out)}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models: List[UNet] = [m.to(self.device).eval() for m in models]
        self.out_channels = self.models[0].out_conv.out_channels

    # -- construction ------------------------------------------------------

    @classmethod
    def from_checkpoints(cls, paths: Iterable[str | Path], *, in_channels: int = 1,
                         out_channels: int = 3, base_features: int = 64,
                         device: Optional[str] = None) -> "EnsembleUNet":
        """Load every checkpoint in *paths* and wrap them."""
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        paths = [Path(p) for p in paths]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"missing checkpoint(s): {missing}")
        models = [UNet.from_checkpoint(p, in_channels=in_channels,
                                       out_channels=out_channels,
                                       base_features=base_features, device=dev)
                  for p in paths]
        return cls(models, device=dev)

    # -- UNet-compatible surface ------------------------------------------

    def eval(self) -> "EnsembleUNet":
        for m in self.models:
            m.eval()
        return self

    def to(self, device: str) -> "EnsembleUNet":
        self.device = device
        self.models = [m.to(device) for m in self.models]
        return self

    def __len__(self) -> int:
        return len(self.models)

    @torch.no_grad()
    def predict_proba(self, image: "np.ndarray", device: Optional[str] = None) -> "np.ndarray":
        """Member-averaged per-class probabilities, shape (C, H, W) float32.

        The same average :meth:`predict` argmaxes -- matches :meth:`UNet.predict_proba`.
        """
        import numpy as np

        dev = device or self.device
        if dev != self.device:
            self.to(dev)
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None]
        x = torch.from_numpy(arr)[None].to(dev)
        act = torch.sigmoid if self.out_channels == 1 else (lambda t: torch.softmax(t, dim=1))
        probs = torch.stack([act(m(x)) for m in self.models]).mean(0)
        return probs[0].cpu().numpy()

    @torch.no_grad()
    def predict(self, image: "np.ndarray", threshold: float = 0.5,
                device: Optional[str] = None, binary_output: bool = False) -> "np.ndarray":
        """Segment one preprocessed frame, averaging across members.

        Matches :meth:`UNet.predict` exactly in signature and return type, so the
        two are interchangeable. Averaging happens on **probabilities**: logits are
        only comparable within one model, and voting on argmaxed masks would throw
        away how confident each member was.
        """
        import numpy as np

        dev = device or self.device
        if dev != self.device:
            self.to(dev)

        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None]
        x = torch.from_numpy(arr)[None].to(dev)

        if self.out_channels == 1:
            probs = torch.stack([torch.sigmoid(m(x)) for m in self.models]).mean(0)
            return (probs[0, 0] >= threshold).to(torch.uint8).cpu().numpy()

        probs = torch.stack([torch.softmax(m(x), dim=1) for m in self.models]).mean(0)
        classes = probs.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
        if binary_output:
            # Collapse to trapped-cell-only before anything downstream sees it --
            # the same contract UNet.predict honours. A decoy blob larger than the
            # cell would otherwise be picked up by MorphologyExtractor.
            return (classes == 1).astype("uint8")
        return classes

    def __repr__(self) -> str:  # pragma: no cover
        return (f"EnsembleUNet({len(self.models)} members, "
                f"out_channels={self.out_channels}, device={self.device!r})")

"""
Unified checkpoint management for PyTorch models (U-Net, MegaNet) and
scikit-learn classifiers (CellClassifier).

CheckpointManager stores artefacts in a versioned sub-directory:

    <root>/
      <name>/
        <name>_best.pt      ← best validation checkpoint (PyTorch)
        <name>_last.pt      ← latest epoch checkpoint
        <name>_v{N}.pt      ← numbered snapshots (when keep_n > 1)
        metadata.yaml       ← training metadata (epoch, metric, timestamp)

For sklearn models the same layout is used but files carry a .pkl extension.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import yaml


class CheckpointManager:
    """Save and load model checkpoints with metadata tracking.

    Parameters
    ----------
    root_dir:
        Base directory under which named sub-directories are created.
    name:
        Model identifier used as sub-directory name and filename prefix.
    keep_n:
        Maximum number of versioned snapshots to retain.  Older ones are
        deleted automatically.  Pass 0 to disable versioned snapshots.
    """

    def __init__(
        self,
        root_dir: Union[str, Path] = "checkpoints",
        name: str = "model",
        keep_n: int = 3,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.name = name
        self.keep_n = keep_n
        self.ckpt_dir = self.root_dir / name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self._version_counter = 0

    # ------------------------------------------------------------------
    # PyTorch model helpers
    # ------------------------------------------------------------------

    def save_torch(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metric: Optional[float] = None,
        is_best: bool = False,
    ) -> Path:
        """Persist a PyTorch model state dict.

        Returns the path of the saved file.
        """
        payload: Dict[str, Any] = {
            "epoch": epoch,
            "metric": metric,
            "timestamp": time.time(),
            "model_state_dict": model.state_dict(),
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()

        # Always overwrite the "last" checkpoint.
        last_path = self.ckpt_dir / f"{self.name}_last.pt"
        torch.save(payload, last_path)

        # Promote to "best" if flagged.
        if is_best:
            best_path = self.ckpt_dir / f"{self.name}_best.pt"
            torch.save(payload, best_path)

        # Versioned snapshot.
        if self.keep_n > 0:
            self._version_counter += 1
            ver_path = self.ckpt_dir / f"{self.name}_v{self._version_counter}.pt"
            torch.save(payload, ver_path)
            self._prune_versions(".pt")

        self._write_metadata(epoch, metric)
        return last_path

    def load_torch(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        tag: str = "best",
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load weights into *model* (and optionally *optimizer*) from a checkpoint.

        Parameters
        ----------
        tag:
            ``"best"`` or ``"last"`` or ``"v{N}"`` for a specific version.
        """
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        path = self.ckpt_dir / f"{self.name}_{tag}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        payload = torch.load(path, map_location=dev)
        model.load_state_dict(payload["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])

        return payload

    # ------------------------------------------------------------------
    # sklearn model helpers
    # ------------------------------------------------------------------

    def save_sklearn(
        self,
        model: Any,
        epoch: int = 0,
        metric: Optional[float] = None,
        is_best: bool = False,
    ) -> Path:
        """Pickle a scikit-learn model."""
        payload = {
            "epoch": epoch,
            "metric": metric,
            "timestamp": time.time(),
            "model": model,
        }
        last_path = self.ckpt_dir / f"{self.name}_last.pkl"
        with open(last_path, "wb") as fh:
            pickle.dump(payload, fh)

        if is_best:
            best_path = self.ckpt_dir / f"{self.name}_best.pkl"
            with open(best_path, "wb") as fh:
                pickle.dump(payload, fh)

        if self.keep_n > 0:
            self._version_counter += 1
            ver_path = self.ckpt_dir / f"{self.name}_v{self._version_counter}.pkl"
            with open(ver_path, "wb") as fh:
                pickle.dump(payload, fh)
            self._prune_versions(".pkl")

        self._write_metadata(epoch, metric)
        return last_path

    def load_sklearn(self, tag: str = "best") -> Any:
        """Restore a pickled sklearn model. Returns the model object."""
        path = self.ckpt_dir / f"{self.name}_{tag}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        return payload["model"]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prune_versions(self, ext: str) -> None:
        """Remove oldest versioned snapshots beyond keep_n."""
        snapshots = sorted(
            self.ckpt_dir.glob(f"{self.name}_v*{ext}"),
            key=lambda p: int(p.stem.split("_v")[-1]),
        )
        while len(snapshots) > self.keep_n:
            snapshots.pop(0).unlink(missing_ok=True)

    def _write_metadata(self, epoch: int, metric: Optional[float]) -> None:
        meta_path = self.ckpt_dir / "metadata.yaml"
        with open(meta_path, "w") as fh:
            yaml.dump(
                {
                    "name": self.name,
                    "last_epoch": epoch,
                    "last_metric": metric,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
                fh,
            )

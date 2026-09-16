"""
Evaluation metrics for segmentation and regression tasks.

SegmentationMetrics
-------------------
  DSC        — Dice Similarity Coefficient  = 2|P∩T| / (|P|+|T|)
  IoU        — Intersection over Union (Jaccard) = |P∩T| / |P∪T|
  Precision  — TP / (TP + FP)
  Recall     — TP / (TP + FN)
  Accuracy   — (TP + TN) / N

All methods accept tensors or numpy arrays and return Python floats / dicts.

RegressionMetrics
-----------------
  MAE  — mean absolute error
  MSE  — mean squared error
  RMSE — root mean squared error
  R²   — coefficient of determination
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
import torch


def _to_flat_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().ravel().astype(np.float32)
    return np.asarray(x, dtype=np.float32).ravel()


class SegmentationMetrics:
    """Pixel-level binary segmentation evaluation.

    Parameters
    ----------
    smooth:
        Laplace smoothing constant added to numerator and denominator to avoid
        division by zero on empty masks.
    threshold:
        Probability cutoff when inputs are soft (float) predictions.
    """

    def __init__(self, smooth: float = 1.0, threshold: float = 0.5) -> None:
        self.smooth = smooth
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def dice(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
    ) -> float:
        p = _to_flat_numpy(pred) >= self.threshold
        t = _to_flat_numpy(target) >= self.threshold
        intersection = (p & t).sum()
        return float((2.0 * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth))

    def iou(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
    ) -> float:
        p = _to_flat_numpy(pred) >= self.threshold
        t = _to_flat_numpy(target) >= self.threshold
        intersection = (p & t).sum()
        union = (p | t).sum()
        return float((intersection + self.smooth) / (union + self.smooth))

    def precision(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
    ) -> float:
        p = _to_flat_numpy(pred) >= self.threshold
        t = _to_flat_numpy(target) >= self.threshold
        tp = (p & t).sum()
        fp = (p & ~t).sum()
        return float((tp + self.smooth) / (tp + fp + self.smooth))

    def recall(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
    ) -> float:
        p = _to_flat_numpy(pred) >= self.threshold
        t = _to_flat_numpy(target) >= self.threshold
        tp = (p & t).sum()
        fn = (~p & t).sum()
        return float((tp + self.smooth) / (tp + fn + self.smooth))

    def accuracy(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
    ) -> float:
        p = _to_flat_numpy(pred) >= self.threshold
        t = _to_flat_numpy(target) >= self.threshold
        correct = (p == t).sum()
        return float(correct / len(p))

    # ------------------------------------------------------------------
    # Batch convenience
    # ------------------------------------------------------------------

    def compute_all(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
    ) -> Dict[str, float]:
        """Compute all metrics and return as a dict.

        Excludes accuracy — with the ~99%+ background-pixel imbalance in
        these frames, per-class accuracy stays trivially near 1.0 even for a
        mediocre segmentation (dominated by correctly-classified background
        pixels), so it doesn't discriminate model quality here the way DSC/
        IoU/precision/recall do. ``accuracy()`` is still available to call
        directly if ever needed.
        """
        return {
            "dsc": self.dice(pred, target),
            "iou": self.iou(pred, target),
            "precision": self.precision(pred, target),
            "recall": self.recall(pred, target),
        }

    # ------------------------------------------------------------------
    # Multiclass convenience
    # ------------------------------------------------------------------

    def per_class(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
        num_classes: int,
        exclude_from_mean: tuple = (0,),
    ) -> Dict[Union[int, str], Dict[str, float]]:
        """Compute all metrics per class for integer-labeled masks.

        Parameters
        ----------
        pred, target:
            Integer class-id arrays/tensors of any shape (e.g. (B, H, W)),
            values in ``[0, num_classes)``.
        num_classes:
            Total number of classes, including background (class 0).
        exclude_from_mean:
            Class ids left out of the ``"mean"`` entry. Defaults to
            ``(0,)`` — background is, by convention throughout this
            codebase, class 0, and its metrics are trivially near-1.0 from
            pixel-count imbalance alone (background is the vast majority of
            pixels in every frame), which would otherwise inflate "mean"
            without saying anything about segmentation quality. Its own
            per-class entry is still computed and returned (callers wanting
            it can read ``result[0]``), just not folded into "mean".

        Returns
        -------
        Dict keyed by class id (each a binary-metrics dict from
        :meth:`compute_all`), plus a ``"mean"`` entry averaging across
        classes not in ``exclude_from_mean``. One-hots each class and
        reuses the existing binary metric implementations, so results for
        class ``c`` match what a binary model trained on class ``c`` alone
        would report.
        """
        pred_np = _to_flat_numpy(pred).astype(np.int64)
        target_np = _to_flat_numpy(target).astype(np.int64)

        per_class_results: Dict[Union[int, str], Dict[str, float]] = {}
        for c in range(num_classes):
            pred_c = (pred_np == c).astype(np.float32)
            target_c = (target_np == c).astype(np.float32)
            per_class_results[c] = self.compute_all(pred_c, target_c)

        mean_classes = [c for c in range(num_classes) if c not in exclude_from_mean]
        mean_result = {
            key: float(np.mean([per_class_results[c][key] for c in mean_classes]))
            for key in ("dsc", "iou", "precision", "recall")
        }
        per_class_results["mean"] = mean_result
        return per_class_results


class RegressionMetrics:
    """Standard scalar regression evaluation metrics."""

    @staticmethod
    def mae(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
        t = np.asarray(y_true, dtype=np.float64)
        p = np.asarray(y_pred, dtype=np.float64)
        return float(np.mean(np.abs(t - p)))

    @staticmethod
    def mse(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
        t = np.asarray(y_true, dtype=np.float64)
        p = np.asarray(y_pred, dtype=np.float64)
        return float(np.mean((t - p) ** 2))

    @staticmethod
    def rmse(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
        t = np.asarray(y_true, dtype=np.float64)
        p = np.asarray(y_pred, dtype=np.float64)
        return float(np.sqrt(np.mean((t - p) ** 2)))

    @staticmethod
    def r2(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
        t = np.asarray(y_true, dtype=np.float64)
        p = np.asarray(y_pred, dtype=np.float64)
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        return float(1.0 - ss_res / (ss_tot + 1e-8))

    def compute_all(
        self,
        y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list],
    ) -> Dict[str, float]:
        return {
            "mae": self.mae(y_true, y_pred),
            "mse": self.mse(y_true, y_pred),
            "rmse": self.rmse(y_true, y_pred),
            "r2": self.r2(y_true, y_pred),
        }

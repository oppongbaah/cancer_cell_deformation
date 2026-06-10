"""
Visualisation utilities for training diagnostics and result inspection.

Visualizer methods
------------------
  plot_training_curves  — loss + DSC vs epoch (segmentation)
  plot_regression_curves — loss + MAE vs epoch (MegaNet)
  plot_mask_overlay     — predicted mask overlaid on the original image
  plot_feature_distributions — box plots of morphological features by class label
  plot_correlation_heatmap   — Pearson correlation matrix of feature columns
  plot_deformation_timeseries — deformation index / aspect ratio vs time with RoD annotation
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Visualizer:
    """Factory class for library-wide visualisation.

    Parameters
    ----------
    figsize:
        Default (width, height) in inches for all figures.
    dpi:
        Output resolution in dots-per-inch.
    style:
        Matplotlib style sheet name (e.g. ``"seaborn-v0_8-whitegrid"``).
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 5),
        dpi: int = 150,
        style: str = "seaborn-v0_8-whitegrid",
    ) -> None:
        self.figsize = figsize
        self.dpi = dpi
        try:
            plt.style.use(style)
        except OSError:
            pass   # fall back to matplotlib default if style unavailable

    # ------------------------------------------------------------------
    # Training curves
    # ------------------------------------------------------------------

    def plot_training_curves(
        self,
        history: Dict[str, list],
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot train/val loss and validation DSC from :meth:`SegmentationTrainer.fit`."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)

        epochs = range(1, len(history["train_loss"]) + 1)
        ax1.plot(epochs, history["train_loss"], label="Train loss")
        ax1.plot(epochs, history["val_loss"], label="Val loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("DiceBCE Loss")
        ax1.set_title("Segmentation Loss")
        ax1.legend()

        ax2.plot(epochs, history["val_dsc"], color="green", label="Val DSC")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("DSC")
        ax2.set_title("Validation Dice Score")
        ax2.legend()

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(Path(save_path))
        return fig

    def plot_regression_curves(
        self,
        history: Dict[str, list],
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot MegaNet train/val loss and MAE from trainer history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)

        epochs = range(1, len(history["train_loss"]) + 1)
        ax1.plot(epochs, history["train_loss"], label="Train loss")
        ax1.plot(epochs, history.get("val_loss", []), label="Val loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("MSE Loss")
        ax1.set_title("Regression Loss (MegaNet)")
        ax1.legend()

        if "val_mae" in history:
            ax2.plot(epochs, history["val_mae"], color="orange", label="Val MAE")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("MAE")
            ax2.set_title("Validation MAE")
            ax2.legend()

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(Path(save_path))
        return fig

    # ------------------------------------------------------------------
    # Segmentation overlay
    # ------------------------------------------------------------------

    def plot_mask_overlay(
        self,
        image: np.ndarray,
        pred_mask: np.ndarray,
        gt_mask: Optional[np.ndarray] = None,
        alpha: float = 0.4,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Overlay predicted (and optionally ground-truth) mask on the image.

        Parameters
        ----------
        image:
            Grayscale float32 array (H × W) in [0, 1] or uint8 (H × W).
        pred_mask:
            Binary uint8 array (H × W), 1 = cell.
        gt_mask:
            Optional ground-truth mask for side-by-side comparison.
        """
        ncols = 3 if gt_mask is not None else 2
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), dpi=self.dpi)

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(image, cmap="gray")
        axes[1].imshow(pred_mask, cmap="Reds", alpha=alpha * pred_mask.astype(float))
        axes[1].set_title("Predicted Mask")
        axes[1].axis("off")

        if gt_mask is not None:
            axes[2].imshow(image, cmap="gray")
            axes[2].imshow(gt_mask, cmap="Greens", alpha=alpha * gt_mask.astype(float))
            axes[2].set_title("Ground-Truth Mask")
            axes[2].axis("off")

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(Path(save_path))
        return fig

    # ------------------------------------------------------------------
    # Feature analysis
    # ------------------------------------------------------------------

    def plot_feature_distributions(
        self,
        features_df: pd.DataFrame,
        label_col: str = "label",
        feature_cols: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Box plots of morphological features grouped by class label."""
        if feature_cols is None:
            feature_cols = [c for c in features_df.columns if c != label_col]

        n = len(feature_cols)
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), dpi=self.dpi)
        axes = np.array(axes).ravel()

        for i, feat in enumerate(feature_cols):
            groups = [
                features_df.loc[features_df[label_col] == lbl, feat].dropna().values
                for lbl in sorted(features_df[label_col].unique())
            ]
            axes[i].boxplot(groups, labels=sorted(features_df[label_col].unique()))
            axes[i].set_title(feat)
            axes[i].set_ylabel("Value")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(Path(save_path))
        return fig

    def plot_correlation_heatmap(
        self,
        features_df: pd.DataFrame,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Pearson correlation matrix for numeric feature columns."""
        corr = features_df.select_dtypes(include=np.number).corr()

        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
        fig.colorbar(im, ax=ax, shrink=0.8)

        ticks = range(len(corr.columns))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(corr.columns, fontsize=8)
        ax.set_title("Feature Correlation Matrix")

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(Path(save_path))
        return fig

    # ------------------------------------------------------------------
    # Deformation time-series
    # ------------------------------------------------------------------

    def plot_deformation_timeseries(
        self,
        timestamps: np.ndarray,
        metric_values: np.ndarray,
        metric_name: str = "deformation_index",
        rod: Optional[float] = None,
        laser_power_mW: Optional[float] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Line plot of a deformation metric over the experimental time axis.

        Annotates the estimated Rate of Deformation (RoD) if provided.
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.plot(timestamps, metric_values, marker="o", markersize=3, linewidth=1.2, label=metric_name)

        if rod is not None:
            label = f"RoD = {rod:.4f} /s"
            # Overlay trend line.
            t0, tf = timestamps[0], timestamps[-1]
            trend = rod * (timestamps - t0) + metric_values[0]
            ax.plot(timestamps, trend, "--", color="red", linewidth=1, label=label)

        title = f"Deformation Time-Series"
        if laser_power_mW is not None:
            title += f"  (P = {laser_power_mW} mW)"
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.legend()

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(Path(save_path))
        return fig

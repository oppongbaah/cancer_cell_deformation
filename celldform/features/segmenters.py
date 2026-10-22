"""
Which segmentation checkpoint segments which cell, for feature extraction.

Every cell's feature frames are segmented by a cross-validation round model that never saw that cell.
Two protocols supply such models; pick one by its tag (``--model`` in scripts/features/extract_features.py,
``--feature-model`` in scripts/report/build_report_assets.py):

    nnunet100_5fold    5-fold, nnU-Net 100 epochs   (checkpoints on the workstation)
    nnunet1000_5fold   5-fold, nnU-Net 1000 epochs  (checkpoints on the cluster)
    nnunet1000_loocv   leave-one-cell-out, nnU-Net 1000 epochs (checkpoints on the cluster)

``plan(tag)`` gives, per round, the nnU-Net fold, and the cells it trained on, validated on and tests
(segments). Read from disk, never typed in:
    5-fold  results/randswap/experiment_0_cv_baseline/cv_baseline_fold_<k>_split.json
    LOOCV   the held-out cell of each round from the run's eval log (results/<exp>/eval_log_native.jsonl),
            its validation cell from the setup job's log (logs/hpc/loocv_nnunet_setup_*.out)
nnU-Net numbers its folds from 0, the rounds from 1: fold = round - 1 in both protocols.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CV_SPLITS = ROOT / "results" / "randswap" / "experiment_0_cv_baseline"
DATASET_5FOLD = "Dataset501_CellTrapRandswap"
LOOCV_SETUP_LOG = "loocv_nnunet_setup_*.out"


@dataclass(frozen=True)
class Segmenter:
    protocol: str       # "5fold" or "loocv"
    exp: str            # experiment folder under checkpoints/ and results/
    dataset: str        # nnU-Net dataset name
    trainer: str        # nnU-Net trainer class
    label: str          # for figures and tables

    @property
    def model_dir(self) -> Path:
        return ROOT / "checkpoints" / self.exp / self.dataset / f"{self.trainer}__nnUNetPlans__2d"


MODELS = {
    "nnunet100_5fold": Segmenter("5fold", "randswap/experiment_1_nnunet", DATASET_5FOLD,
                                 "nnUNetTrainer_100epochs", "nnU-Net, 100 epochs, 5-fold"),
    "nnunet1000_5fold": Segmenter("5fold", "randswap/experiment_1b_nnunet_1000ep", DATASET_5FOLD,
                                  "nnUNetTrainer", "nnU-Net, 1000 epochs, 5-fold"),
    "nnunet1000_loocv": Segmenter("loocv", "randswap/experiment_4_loocv/nnunet", "Dataset502_CellTrapLOOCV",
                                  "nnUNetTrainer", "nnU-Net, 1000 epochs, leave-one-cell-out"),
}


def five_fold_records() -> dict[int, dict]:
    """5-fold round k -> its split record (cells per role, HER2, test frames and their hand masks)."""
    out = {}
    for p in sorted(CV_SPLITS.glob("cv_baseline_fold_[0-9]*_split.json")):
        r = json.loads(p.read_text())
        out[int(r["fold"])] = r
    return dict(sorted(out.items()))


def her2() -> dict[str, str]:
    return next(iter(five_fold_records().values()))["her2"]


def plan(tag: str) -> list[dict]:
    """One dict per round: round, fold, train, val, test (cells). The test cells are the ones that
    round's checkpoint segments."""
    seg = MODELS[tag]
    cells = sorted(her2())
    if seg.protocol == "5fold":
        rounds = [{"round": k, "fold": k - 1, **{role: list(r["cells"][role]) for role in ("train", "val", "test")}}
                  for k, r in five_fold_records().items()]
    else:
        test = {}
        for line in (ROOT / "results" / seg.exp / "eval_log_native.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["filter"]:
                k = int(r["fold"])
                if f"fold_{k - 1}/" not in r["checkpoint"]:
                    raise ValueError(f"LOOCV eval log: round {k} was not scored with fold_{k - 1}")
                test[k] = r["filter"]
        logs = sorted((ROOT / "logs" / "hpc").glob(LOOCV_SETUP_LOG))
        if not logs:
            raise FileNotFoundError(f"logs/hpc/{LOOCV_SETUP_LOG} -- the LOOCV setup log names each round's validation cell")
        val = {}
        for k, v, t in re.findall(r"round (\d+) \(nnU-Net fold \d+\):.*?val \d+ \((\S+)\)\s+test \d+ \((\S+)\)",
                                  logs[-1].read_text()):
            if test.get(int(k)) != t:
                raise ValueError(f"LOOCV round {k}: setup log tests {t}, eval log scored {test.get(int(k))}")
            val[int(k)] = v
        rounds = [{"round": k, "fold": k - 1, "val": [val[k]], "test": [t],
                   "train": [c for c in cells if c not in (t, val[k])]} for k, t in sorted(test.items())]
    tested = sorted(c for r in rounds for c in r["test"])
    if tested != cells:
        raise ValueError(f"{tag}: the rounds do not test each of the {len(cells)} cells exactly once")
    return rounds


def round_of(tag: str) -> dict[str, int]:
    """cell -> the round whose checkpoint segments it."""
    return {c: r["round"] for r in plan(tag) for c in r["test"]}

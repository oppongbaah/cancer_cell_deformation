"""
Dataset organisation for optical-tweezer cancer cell experiments.

Reads from data/raw/ and writes four pipeline-ready folders under data/frames/:

  01_annotate_pool/   Legacy images + 2025-camera domain-adapt samples → annotate
  02_unet_holdout/    Held-out cells for U-Net evaluation
  03_mask_factory/    Bulk labeled frames → U-Net inference → classifier features
  04_clf_arena/       Held-out cells for final classifier evaluation

Video cell assignments
──────────────────────
  U-Net test    : high-9,  low-6
  Classifier test: high-10, low-5
  Inference / clf-train: high 1–4,6–8  +  low 1–4
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path
from typing import List, Tuple

from celldform.acquisition.extractor import FrameExtractor

DEFAULT_RAW_DIR    = Path("data/raw")
DEFAULT_FRAMES_DIR = Path("data/frames")
DEFAULT_EVERY_N    = 200
DEFAULT_DA_N       = 5
TARGET_SIZE        = (1228, 922)


# ---------------------------------------------------------------------------
# Video assignment
# ---------------------------------------------------------------------------

def _video_lists(raw: Path) -> dict[str, List[Tuple[Path, str, str]]]:
    high = raw / "high"
    low  = raw / "low"
    return {
        "unet_test": [
            (high / "cell 9 high 50 increments.mp4",  "high", "cell9"),
            (low  / "cell 6 low 50 increments.mp4",   "low",  "cell6"),
        ],
        "clf_test": [
            (high / "cell 10 high 50 increments.mp4", "high", "cell10"),
            (low  / "cell 5 low 50 increments.mp4",   "low",  "cell5"),
        ],
        "inference": [
            (high / "cell 1 high 50 increments.mp4",  "high", "cell1"),
            (high / "cell 2 high 50 increments.mp4",  "high", "cell2"),
            (high / "cell 3 high 50 increments.mp4",  "high", "cell3"),
            (high / "cell 4 high 50 increments.mp4",  "high", "cell4"),
            (high / "cell 6 high 50 increments.mp4",  "high", "cell6"),
            (high / "cell 7 high 50 increments.mp4",  "high", "cell7"),
            (high / "cell 8 high 50 increments.mp4",  "high", "cell8"),
            (raw  / "cell 1 low 50 increments.mp4",   "low",  "cell1"),
            (low  / "cell 2 low 50 increments.mp4",   "low",  "cell2"),
            (low  / "cell 3 low 50 increments.mp4",   "low",  "cell3"),
            (low  / "cell 4 low 50 increments.mp4",   "low",  "cell4"),
        ],
        "domain_adapt": [
            (high / "cell 1 high 50 increments.mp4",  "high", "cell1"),
            (high / "cell 2 high 50 increments.mp4",  "high", "cell2"),
            (high / "cell 3 high 50 increments.mp4",  "high", "cell3"),
            (high / "cell 4 high 50 increments.mp4",  "high", "cell4"),
            (high / "cell 6 high 50 increments.mp4",  "high", "cell6"),
            (high / "cell 7 high 50 increments.mp4",  "high", "cell7"),
            (high / "cell 8 high 50 increments.mp4",  "high", "cell8"),
            (raw  / "cell 1 low 50 increments.mp4",   "low",  "cell1"),
            (low  / "cell 2 low 50 increments.mp4",   "low",  "cell2"),
            (low  / "cell 3 low 50 increments.mp4",   "low",  "cell3"),
            (low  / "cell 4 low 50 increments.mp4",   "low",  "cell4"),
        ],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cap_frame_count(path: Path) -> int:
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _extract_group(
    video_list: List[Tuple[Path, str, str]],
    out_dir: Path,
    every_n: int,
    target_size: Tuple[int, int],
) -> List[dict]:
    """Extract every_n-th frame from each video into out_dir via seek."""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[dict] = []

    for vid_path, her2, cell_id in video_list:
        if not vid_path.exists():
            print(f"  [SKIP] {vid_path.name}", flush=True)
            continue

        total   = _cap_frame_count(vid_path)
        targets = list(range(0, total, every_n))

        FrameExtractor(
            vid_path,
            output_dir=out_dir,
            target_size=target_size,
            image_format="jpg",
            change_threshold=None,
        ).seek_extract(targets, file_stem=f"{her2}_{cell_id}")

        for t in targets:
            p = out_dir / f"{her2}_{cell_id}_{t:06d}.jpg"
            if p.exists():
                all_rows.append({
                    "source":      str(vid_path),
                    "her2":        her2,
                    "cell_id":     cell_id,
                    "frame_index": t,
                    "output_path": str(p),
                })
        print(f"  {her2}/{cell_id}: {len(targets)} frames", flush=True)

    _write_csv(out_dir / "manifest.csv", all_rows)
    return all_rows


def _domain_adapt(
    video_list: List[Tuple[Path, str, str]],
    ann_dir: Path,
    n: int,
    target_size: Tuple[int, int],
) -> List[dict]:
    """Copy n evenly-spaced frames from each 2025 video into the annotation pool."""
    rows: List[dict] = []
    for vid_path, her2, cell_id in video_list:
        if not vid_path.exists():
            print(f"  [SKIP domain] {vid_path.name}", flush=True)
            continue

        total   = _cap_frame_count(vid_path)
        step    = max(1, total // n)
        targets = [i * step for i in range(n)]

        FrameExtractor(
            vid_path,
            output_dir=ann_dir,
            target_size=target_size,
            image_format="jpg",
            change_threshold=None,
        ).seek_extract(targets, file_stem=f"domain_{her2}_{cell_id}")

        for t in targets:
            p = ann_dir / f"domain_{her2}_{cell_id}_{t:06d}.jpg"
            if p.exists():
                rows.append({
                    "source":      str(vid_path),
                    "her2":        her2,
                    "cell_id":     cell_id,
                    "frame_index": t,
                    "output_path": str(p),
                })
        print(f"  domain/{her2}/{cell_id}: {len(targets)} frames", flush=True)

    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def organise(
    raw_dir: Path = DEFAULT_RAW_DIR,
    frames_dir: Path = DEFAULT_FRAMES_DIR,
    every_n: int = DEFAULT_EVERY_N,
    domain_adapt_n: int = DEFAULT_DA_N,
    clean: bool = True,
) -> None:
    """Organise raw data into the four pipeline-ready frame folders."""
    t0     = time.time()
    videos = _video_lists(raw_dir)

    folders = {
        "annotate":  frames_dir / "01_annotate_pool",
        "unet_test": frames_dir / "02_unet_holdout",
        "inference": frames_dir / "03_mask_factory",
        "clf_test":  frames_dir / "04_clf_arena",
    }

    if clean and frames_dir.exists():
        print(f"Cleaning {frames_dir} ...", flush=True)
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Legacy cancer images
    ann_dir   = folders["annotate"]
    ann_dir.mkdir(parents=True, exist_ok=True)
    legacy_dir = raw_dir / "legacy/20210825_cancer"
    print("\n[annotate_pool] Copying legacy cancer images ...", flush=True)
    ann_rows: List[dict] = []
    for img in sorted(legacy_dir.glob("*.jpg")) + sorted(legacy_dir.glob("*.JPG")):
        shutil.copy(img, ann_dir / img.name)
        ann_rows.append({
            "source":      str(img),
            "her2":        "unlabeled",
            "cell_id":     "legacy",
            "frame_index": "",
            "output_path": str(ann_dir / img.name),
        })
    print(f"  → {len(ann_rows)} images", flush=True)

    # Domain-adapt samples
    print("\n[annotate_pool] Adding 2025-camera domain samples ...", flush=True)
    da_rows = _domain_adapt(videos["domain_adapt"], ann_dir, domain_adapt_n, TARGET_SIZE)
    ann_rows.extend(da_rows)
    _write_csv(ann_dir / "manifest.csv", ann_rows)
    print(f"  → {len(da_rows)} domain frames  ({len(ann_rows)} total in pool)", flush=True)

    # U-Net test hold-out
    print(f"\n[unet_holdout] → {folders['unet_test'].name}", flush=True)
    rows = _extract_group(videos["unet_test"], folders["unet_test"], every_n, TARGET_SIZE)
    print(f"  → {len(rows)} frames total", flush=True)

    # Mask factory
    print(f"\n[mask_factory] → {folders['inference'].name}", flush=True)
    rows = _extract_group(videos["inference"], folders["inference"], every_n, TARGET_SIZE)
    print(f"  → {len(rows)} frames total", flush=True)

    # Classifier arena
    print(f"\n[clf_arena] → {folders['clf_test'].name}", flush=True)
    rows = _extract_group(videos["clf_test"], folders["clf_test"], every_n, TARGET_SIZE)
    print(f"  → {len(rows)} frames total", flush=True)

    # Summary
    print(f"\n{'─'*60}", flush=True)
    print("Dataset layout:", flush=True)
    for folder in folders.values():
        imgs = list(folder.glob("*.jpg")) + list(folder.glob("*.JPG"))
        print(f"  {folder.name:<32} {len(imgs):>5} images", flush=True)
    print(f"\nDone in {time.time() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="celldform-organize-dataset",
        description="Organise raw optical-tweezer data into pipeline-ready frame folders.",
    )
    p.add_argument(
        "--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
        help=f"Root directory of raw data (default: {DEFAULT_RAW_DIR}).",
    )
    p.add_argument(
        "--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR,
        help=f"Output root for organised frames (default: {DEFAULT_FRAMES_DIR}).",
    )
    p.add_argument(
        "--every-n", type=int, default=DEFAULT_EVERY_N,
        help=f"Sample every N-th frame from each video (default: {DEFAULT_EVERY_N}).",
    )
    p.add_argument(
        "--domain-adapt-n", type=int, default=DEFAULT_DA_N,
        help=f"Frames per cell added to annotate_pool for domain adaptation (default: {DEFAULT_DA_N}).",
    )
    p.add_argument(
        "--no-clean", action="store_true",
        help="Skip wiping frames-dir before running (useful for partial re-runs).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not args.raw_dir.exists():
        print(f"[ERROR] --raw-dir not found: {args.raw_dir}", file=sys.stderr)
        sys.exit(1)
    organise(
        raw_dir=args.raw_dir,
        frames_dir=args.frames_dir,
        every_n=args.every_n,
        domain_adapt_n=args.domain_adapt_n,
        clean=not args.no_clean,
    )

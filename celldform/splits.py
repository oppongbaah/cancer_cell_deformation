"""
Cell-level cross-validation splits (randswap, 2026-10-01).

Supervisor feedback: the testing cells must be drawn at random, not chosen by hand.
Everything here is driven by one config section, :class:`celldform.config.SplitCfg`:

    split:
      train_cells: 11      # whole cells, never frames
      val_cells: 1
      test_cells: 3        # -> roster / 3 = 5 folds; every cell tested exactly once
      cell_split_seed: 42
      stratify_by: her2    # every test fold holds both classes

The draw is deterministic: the same roster and seed always give the same folds.

How the folds are made
----------------------
Each HER2 class is shuffled on its own (seeded), the two lists are concatenated (high
then low) and dealt round-robin into the folds. With 9 high and 6 low cells and 5 folds
that gives four folds of 2 high + 1 low and one of 1 high + 2 low -- stratified, yet
random within each class. The validation cell(s) of each fold are then drawn from that
fold's non-test cells with a fold-specific seed.

Legacy (2021) frames have no cell and no HER2 label: they always train.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_EXTS = (".jpg", ".jpeg", ".png")
_CELL = re.compile(r"^(?:domain_)?(high|low)_(cell\d+)_(\d+)$")


@dataclass(frozen=True)
class Frame:
    """One annotated frame. ``cell`` is e.g. ``high_cell2``; None for legacy frames."""
    stem: str
    image: Path
    mask: Path
    cell: Optional[str]
    her2: Optional[str]
    index: int = -1          # frame index in its video (orders a cell's frames in time)


@dataclass
class FoldAssignment:
    fold: str                              # "1".."k" or "final"
    train: List[str]
    val: List[str]
    test: List[str]
    roster: Dict[str, str] = field(default_factory=dict)    # cell -> her2

    def role(self, cell: Optional[str]) -> str:
        if cell is None or cell in self.train:
            return "train"
        return "val" if cell in self.val else "test"


def parse_stem(stem: str) -> tuple[Optional[str], Optional[str], int]:
    """(cell, her2, frame index) from a frame name; (None, None, -1) for legacy frames."""
    m = _CELL.match(stem)
    if not m:
        return None, None, -1
    return f"{m.group(1)}_{m.group(2)}", m.group(1), int(m.group(3))


def collect_frames(sources: Sequence[Sequence[str]]) -> List[Frame]:
    """Every annotated frame across ``sources`` ([frames_dir, masks_dir] pairs).

    A frame counts only if its mask exists. A stem appearing in two sources is an error:
    the same frame must not be counted twice.
    """
    out: Dict[str, Frame] = {}
    for frames_dir, masks_dir in sources:
        for img in sorted(Path(frames_dir).iterdir()):
            if img.suffix.lower() not in _EXTS or not img.is_file():
                continue
            mask = Path(masks_dir) / f"{img.stem}.png"
            if not mask.exists():
                continue
            if img.stem in out:
                raise ValueError(f"frame {img.stem} appears in more than one source")
            cell, her2, idx = parse_stem(img.stem)
            if cell is None and not img.stem.isdigit():
                raise ValueError(f"cannot tell which cell {img.name} belongs to")
            out[img.stem] = Frame(img.stem, img, mask, cell, her2, idx)
    return sorted(out.values(), key=lambda f: f.stem)


def roster_of(frames: Sequence[Frame]) -> Dict[str, str]:
    """cell -> HER2 class, for every real cell present."""
    return {f.cell: f.her2 for f in frames if f.cell is not None}


def make_folds(roster: Dict[str, str], cfg) -> List[List[str]]:
    """The test cells of each fold. Validates the counts against the roster first."""
    n = len(roster)
    if cfg.train_cells + cfg.val_cells + cfg.test_cells != n:
        raise ValueError(f"split counts {cfg.train_cells}+{cfg.val_cells}+{cfg.test_cells} "
                         f"!= {n} cells on disk")
    if n % cfg.test_cells:
        raise ValueError(f"test_cells={cfg.test_cells} does not divide {n} cells into equal folds")
    k = n // cfg.test_cells
    rng = random.Random(cfg.cell_split_seed)
    if cfg.stratify_by:
        order: List[str] = []
        for cls in sorted(set(roster.values())):
            members = sorted(c for c, h in roster.items() if h == cls)
            rng.shuffle(members)
            order += members
    else:
        order = sorted(roster)
        rng.shuffle(order)
    folds: List[List[str]] = [[] for _ in range(k)]
    for i, cell in enumerate(order):
        folds[i % k].append(cell)
    if cfg.stratify_by:
        classes = set(roster.values())
        for i, f in enumerate(folds, 1):
            if {roster[c] for c in f} != classes:
                raise ValueError(f"fold {i} does not contain every {cfg.stratify_by} class: {f}")
    return [sorted(f) for f in folds]


def assign(roster: Dict[str, str], cfg, fold: str) -> FoldAssignment:
    """Train / val / test cells for ``fold`` ("1".."k", or "final" = every cell trains)."""
    folds = make_folds(roster, cfg)
    if fold == "final":
        return FoldAssignment("final", sorted(roster), [], [], dict(roster))
    k = int(fold)
    if not 1 <= k <= len(folds):
        raise ValueError(f"fold must be 1..{len(folds)} or 'final', got {fold}")
    test = folds[k - 1]
    rest = sorted(c for c in roster if c not in test)
    val = sorted(random.Random(f"{cfg.cell_split_seed}-val-{k}").sample(rest, cfg.val_cells))
    train = [c for c in rest if c not in val]
    a = FoldAssignment(str(k), train, val, test, dict(roster))
    assert len(a.train) == cfg.train_cells and not (set(a.train) & set(a.val) | set(a.train) & set(a.test)
                                                     | set(a.val) & set(a.test))
    return a


def pick_test_frames(frames: Sequence[Frame], cell: str, n: int) -> List[Frame]:
    """``n`` frames of ``cell``, evenly spaced through the recording (by frame index)."""
    own = sorted((f for f in frames if f.cell == cell), key=lambda f: f.index)
    if len(own) <= n:
        return own
    picks = np.unique(np.round(np.linspace(0, len(own) - 1, n)).astype(int))
    return [own[i] for i in picks]


def describe(a: FoldAssignment, frames: Sequence[Frame]) -> str:
    """Human-readable split summary with cell and frame percentages."""
    n_cells = len(a.roster)
    per_cell = {c: sum(1 for f in frames if f.cell == c) for c in a.roster}
    n_legacy = sum(1 for f in frames if f.cell is None)
    lines = [f"fold {a.fold}:"]
    for role, cells in (("train", a.train), ("val", a.val), ("test", a.test)):
        nf = sum(per_cell[c] for c in cells) + (n_legacy if role == "train" else 0)
        tag = ", ".join(f"{c.replace('_cell', '-')}({per_cell[c]})" for c in cells) or "-"
        lines.append(f"  {role:5s} {len(cells):2d} cells ({100 * len(cells) / n_cells:4.1f}%)"
                     f"  {nf:4d} frames{' incl. ' + str(n_legacy) + ' legacy' if role == 'train' else ''}"
                     f"  {tag}")
    return "\n".join(lines)

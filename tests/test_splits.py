"""Tests for celldform.splits (randswap cell-level folds).

These use a SYNTHETIC roster on purpose: running make_folds on the real cells is the
draw itself, which must only happen after the config is committed.
"""

import pytest

from celldform.config import SplitCfg
from celldform.splits import Frame, assign, make_folds, parse_stem, pick_test_frames

ROSTER = {**{f"high_cell{i}": "high" for i in range(1, 10)},
          **{f"low_cell{i}": "low" for i in range(1, 7)}}          # 9 high + 6 low, like ours


def cfg(**kw):
    return SplitCfg(**{"train_cells": 11, "val_cells": 1, "test_cells": 3, **kw})


def test_every_cell_tested_exactly_once_and_folds_are_stratified():
    folds = make_folds(ROSTER, cfg())
    assert len(folds) == 5 and all(len(f) == 3 for f in folds)
    tested = [c for f in folds for c in f]
    assert sorted(tested) == sorted(ROSTER)                          # once each
    for f in folds:
        assert {ROSTER[c] for c in f} == {"high", "low"}            # both classes
    mix = sorted(sum(ROSTER[c] == "high" for c in f) for f in folds)
    assert mix == [1, 2, 2, 2, 2]


def test_draw_is_deterministic_and_depends_on_the_seed():
    assert make_folds(ROSTER, cfg()) == make_folds(ROSTER, cfg())
    assert make_folds(ROSTER, cfg()) != make_folds(ROSTER, cfg(cell_split_seed=7))


def test_assignment_partitions_the_roster():
    for k in range(1, 6):
        a = assign(ROSTER, cfg(), str(k))
        assert (len(a.train), len(a.val), len(a.test)) == (11, 1, 3)
        assert sorted(a.train + a.val + a.test) == sorted(ROSTER)
    final = assign(ROSTER, cfg(), "final")
    assert sorted(final.train) == sorted(ROSTER) and not final.val and not final.test


@pytest.mark.parametrize("kw", [{"train_cells": 10}, {"test_cells": 4, "train_cells": 10}])
def test_bad_counts_are_refused(kw):
    with pytest.raises(ValueError):
        make_folds(ROSTER, cfg(**kw))


def test_stem_parsing_and_even_test_frames():
    assert parse_stem("domain_high_cell2_000498") == ("high_cell2", "high", 498)
    assert parse_stem("low_cell6_005000") == ("low_cell6", "low", 5000)
    assert parse_stem("20210825164050") == (None, None, -1)
    frames = [Frame(f"high_cell1_{i:06d}", None, None, "high_cell1", "high", i) for i in range(0, 10000, 200)]
    picked = pick_test_frames(frames, "high_cell1", 20)
    assert len(picked) == 20 and picked[0].index == 0 and picked[-1].index == 9800

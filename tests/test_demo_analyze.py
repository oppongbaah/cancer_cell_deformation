"""Tests for celldform.demo.analyze (stage 4)."""
import cv2
import numpy as np
import pytest

from celldform.demo.analyze import FrameAnalyzer
from celldform.demo.segment import Seg

H, W = 922, 1228


def seg_of(*discs, decoy=()):
    lab = np.zeros((H, W), np.uint8)
    for (x, y, r) in discs:
        cv2.circle(lab, (x, y), r, 1, -1)
    for (x, y, r) in decoy:
        cv2.circle(lab, (x, y), r, 2, -1)
    return Seg(lab, np.full((H, W), 0.99, np.float32), None)


def test_no_cell_is_reported_not_imputed():
    r = FrameAnalyzer().analyze(seg_of())
    assert r.region is None and not r.passed and r.row["found"] is False
    assert np.isnan(r.row["area"]) and "no cell" in r.row["reasons"]


def test_clean_cell_passes_with_features():
    a = FrameAnalyzer()
    r = a.analyze(seg_of((600, 450, 16)))
    assert r.passed and r.row["area"] == pytest.approx(np.pi * 16 ** 2, rel=0.05)
    assert r.row["len_par"] > 0 and r.row["aspect_ratio"] == pytest.approx(1.0, abs=0.1)


def test_trap_is_learned_and_nearest_region_wins_over_bigger_distractor():
    a = FrameAnalyzer()
    for i in range(4):
        a.analyze(seg_of((600 + i, 450, 16)), i)
    r = a.analyze(seg_of((602, 450, 16), (900, 300, 30)), 4)       # a bigger blob far away
    assert r.row["n_regions"] == 2 and abs(r.row["trap_dist"]) < 10 and r.passed
    assert r.region[450, 602] and not r.region[300, 900]


def test_far_cell_fails_distance_gate():
    a = FrameAnalyzer()
    for i in range(4):
        a.analyze(seg_of((600, 450, 16)), i)
    r = a.analyze(seg_of((700, 450, 16)), 4)       # only region, 100 px from the trap
    assert not r.passed and "from the trap" in r.row["reasons"]


def test_merged_neighbour_fails_touches_and_solidity():
    a = FrameAnalyzer()
    for i in range(4):
        a.analyze(seg_of((600, 450, 16)), i)
    r = a.analyze(seg_of((600, 450, 16), (628, 450, 16)), 4)       # two discs fused: peanut shape
    assert not r.passed


def test_edge_cut_cell_fails():
    r = FrameAnalyzer().analyze(seg_of((3, 450, 16)))
    assert not r.passed and "frame edge" in r.row["reasons"]


def test_typical_area_is_causal_and_catches_a_swollen_mask():
    a = FrameAnalyzer()
    for i in range(5):
        assert a.analyze(seg_of((600, 450, 16)), i).passed
    assert a.typical_area() == pytest.approx(np.pi * 256, rel=0.05)
    r = a.analyze(seg_of((600, 450, 24)), 5)                       # area x2.25
    assert not r.passed and "typical" in r.row["reasons"]
    assert a.typical_area() == pytest.approx(np.pi * 256, rel=0.05)  # a failed frame does not teach it


def test_low_confidence_boundary_fails():
    s = seg_of((600, 450, 16))
    s.maxprob[:] = 0.7
    a = FrameAnalyzer()
    assert "boundary confidence" in a.analyze(s).row["reasons"]


def test_user_set_trap_wins_over_a_wrong_first_frame():
    a = FrameAnalyzer(trap_xy=(600, 450))
    r = a.analyze(seg_of((600, 450, 16), (900, 300, 30)))          # first frame: bigger blob far away
    assert r.passed and r.region[450, 600] and not r.region[300, 900]
    assert a.trap == (600, 450)
    a.analyze(seg_of((900, 300, 30)))                              # later frames never move a fixed trap
    assert a.trap == (600, 450)

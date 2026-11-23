"""Tests for celldform.features.quality -- region selection, segmentation checks, gates, image quality."""
import cv2
import numpy as np
import pytest

from celldform.features.quality import (Gates, apply_gates, image_quality, regions, segmentation_checks,
                                        select_trapped_region, trap_position)


def disc(shape, centre, r, value=1):
    m = np.zeros(shape, np.uint8)
    cv2.circle(m, centre, r, value, -1)
    return m


def test_select_picks_region_nearest_trap_not_largest():
    # a small trapped cell at the trap and a LARGER neighbour further away (the high-10 case)
    m = disc((200, 300), (100, 100), 12) | disc((200, 300), (160, 100), 20)
    reg, n = select_trapped_region(m, trap_xy=(102, 99))
    assert n == 2
    assert abs(reg["centroid"][0] - 100) < 1 and reg["area"] < 600


def test_trap_position_is_robust_median():
    assert trap_position([(10, 10), (11, 10), (12, 11), (90, 90)]) == (11.5, 10.5)
    assert trap_position([]) is None


def test_regions_drops_small_blobs():
    m = disc((100, 100), (50, 50), 10) | disc((100, 100), (10, 10), 2)
    assert len(regions(m, min_area_px=50)) == 1


def test_checks_and_gates_pass_a_clean_round_cell():
    m = disc((200, 200), (100, 100), 15)
    reg, n = select_trapped_region(m, (100, 100))
    c = segmentation_checks(reg, m, np.ones(m.shape, np.float32), typical_area=reg["area"], n_regions=n)
    ok, why = apply_gates(c, Gates())
    assert ok, why
    assert c["solidity"] > 0.95 and c["area_ratio"] == pytest.approx(1.0)


def test_gates_catch_merged_border_and_unsure_masks():
    # two touching discs merged into one region: low solidity
    merged = disc((200, 200), (90, 100), 15) | disc((200, 200), (118, 100), 15)
    reg, n = select_trapped_region(merged, (100, 100))
    c = segmentation_checks(reg, merged, np.full(merged.shape, 0.6, np.float32), typical_area=reg["area"] / 2, n_regions=n)
    ok, why = apply_gates(c, Gates())
    assert not ok
    text = " ".join(why)
    assert "area" in text and "boundary confidence" in text
    # a cell cut by the frame edge
    edge = disc((200, 200), (5, 100), 15)
    reg, n = select_trapped_region(edge, (5, 100))
    assert not apply_gates(segmentation_checks(reg, edge, None, reg["area"], n), Gates())[0]


def test_no_region_fails():
    c = segmentation_checks(None, np.zeros((50, 50), np.uint8), None, None, 0)
    assert apply_gates(c, Gates()) == (False, ["no cell near the trap"])


def test_image_quality_sees_blur_and_brightness():
    rng = np.random.default_rng(0)
    sharp = (rng.random((200, 200)) * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 3)
    assert image_quality(sharp)["focus_frame"] > image_quality(blurred)["focus_frame"]
    dark, bright = np.full((100, 100), 60, np.uint8), np.full((100, 100), 180, np.uint8)
    assert image_quality(bright)["illum_mean"] > image_quality(dark)["illum_mean"]
    cell = disc((100, 100), (50, 50), 10).astype(bool)
    img = np.full((100, 100), 200, np.uint8)
    img[cell] = 100
    assert image_quality(img, cell)["contrast_cell"] == pytest.approx(0.5, abs=0.01)

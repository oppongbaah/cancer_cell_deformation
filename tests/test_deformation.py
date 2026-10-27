import numpy as np
import cv2
import pandas as pd

from celldform.features.deformation import cell_summary, frame_features, theil_sen, within_step


def ellipse(a: int, b: int, angle: float = 0.0, size: int = 200) -> np.ndarray:
    m = np.zeros((size, size), np.uint8)
    cv2.ellipse(m, (size // 2, size // 2), (a, b), angle, 0, 360, 1, -1)
    return m.astype(bool)


def test_lengths_follow_the_axes():
    f = frame_features(ellipse(40, 25))
    assert abs(f["len_par"] - 80) < 3 and abs(f["len_perp"] - 50) < 3      # 4 sigma = full length
    assert abs(f["orientation"]) < 2


def test_rotated_ellipse_turns_orientation_and_swaps_lengths():
    f = frame_features(ellipse(40, 25, angle=90))
    assert f["len_perp"] > f["len_par"]
    assert abs(abs(f["orientation"]) - 90) < 2
    assert abs(abs(f["ellip_phase"]) - 90) < 5


def test_circle_has_no_elongation_or_roughness():
    f = frame_features(ellipse(30, 30))
    assert f["ellip_f2"] < 0.02 and f["roughness"] < 0.05
    e = frame_features(ellipse(40, 25))
    assert e["ellip_f2"] > 0.2 and abs(e["ellip_phase"]) < 5


def test_empty_mask_gives_nan():
    assert all(np.isnan(v) for v in frame_features(np.zeros((50, 50), bool)).values())


def test_theil_sen_ignores_an_outlier():
    x = np.arange(20.0)
    y = 2 * x
    y[5] = 500
    assert abs(theil_sen(x, y) - 2) < 1e-9


def test_cell_summary_recovers_a_linear_stretch():
    steps = np.arange(50)
    lp = 100 * (1 + 0.002 * steps)              # +0.2 % per step along the beam
    lq = 80 * (1 - 0.001 * steps)               # -0.1 % per step across it
    df = pd.DataFrame({"laser_step": steps, "passed": True, "len_par": lp, "len_perp": lq,
                       "volume": np.pi / 6 * lp * lq ** 2, "ellip_f2": 0.1, "roughness": 0.02, "orientation": 0.0})
    s = cell_summary(df)
    assert abs(s["compliance_par"] - 0.002 * 100 / 104.5) < 2e-4   # relative to the steps-0..9 baseline
    assert abs(s["poisson"] - 0.5 * 80 / 79.64 * 104.5 / 100) < 0.05
    assert s["orient_spread"] < 1e-6


def test_within_step_fits_a_creep():
    rows = []
    for s in range(20):
        for o in (20, 60, 100, 140, 180):
            t = o / 22.0
            rows.append({"laser_step": s, "frame_index": 200 * s + o, "passed": True,
                         "len_par": 100 * (1 + 0.01 * s) * (1 + 0.02 * (1 - np.exp(-t / 2.0)))})
    w = within_step(pd.DataFrame(rows))
    assert w["creep"] > 0.01
    assert abs(w["creep_tau"] - 2.0) < 0.3

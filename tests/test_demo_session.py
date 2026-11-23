"""Tests for celldform.demo.framework and .session (stage 5)."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from celldform.demo import framework as fw
from celldform.demo.segment import Seg
from celldform.demo.session import CellSession

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "checkpoints/classification/nnunet1000_loocv/framework/her2_framework.joblib"
H, W = 922, 1228


def seg(r=16, x=600, y=450, ax=1.0):
    lab = np.zeros((H, W), np.uint8)
    cv2.ellipse(lab, (x, y), (int(r * ax), r), 0, 0, 360, 1, -1)
    return Seg(lab, np.full((H, W), 0.99, np.float32), None)


def fake_framework():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    steps = pd.DataFrame(rng.normal(size=(60, len(fw.STEP_FEATURES))), columns=fw.STEP_FEATURES)
    steps["y"] = rng.integers(0, 2, 60)
    cells = pd.DataFrame(rng.normal(size=(15, 4)), columns=fw.SELECTED)
    cells["y"] = [0, 1] * 7 + [0]
    return fw.fit(steps, cells, LogisticRegression())


def feed(session, n_frames, **kw):
    for i in range(n_frames):
        if session.wants(i):
            session.add(seg(**kw), i)


def test_sampling_grid():
    s = CellSession()
    assert [i for i in range(400) if s.wants(i)] == [0, 20, 60, 100, 140, 180, 200, 220, 260, 300, 340, 380]
    assert s.step_of(199) == 0 and s.step_of(200) == 1


def test_empty_and_short_session_withholds_verdict():
    s = CellSession(fake_framework())
    v = s.verdict()
    assert v.status == "insufficient" and "no frames" in v.reason
    feed(s, 2200 - 1)                                   # 11 steps' worth of frames minus one
    feed_steps = s.steps()
    assert len(feed_steps) == 11
    s2 = CellSession(fake_framework())
    feed(s2, 2 * 200)
    v2 = s2.verdict()
    assert v2.status == "insufficient" and v2.label is None and "at least 3" in v2.reason and v2.n_steps == 2


def test_no_classifier_is_reported():
    s = CellSession(None)
    feed(s, 2400)
    assert "no classifier" in s.verdict().reason


def test_low_pass_fraction_withholds_verdict():
    s = CellSession(fake_framework())
    k = 0
    for i in range(2400):
        if s.wants(i):
            s.add(seg(x=100 if k % 4 == 3 else 600), i)     # every 4th measured frame jumps 500 px
            k += 1
    v = s.verdict()
    assert v.status == "insufficient" and "pass the checks" in v.reason


def test_full_session_gives_verdict_and_caveats():
    s = CellSession(fake_framework(), caveats=["laser protocol differs"])
    for i in range(2600):
        if s.wants(i):
            s.add(seg(r=16, ax=1.0 + 0.02 * (i // 200)), i)      # cell lengthens a little each step
    v = s.verdict()
    assert v.status == "ok" and v.label in ("high", "low") and 0 <= v.p_high <= 1
    assert v.p_high == pytest.approx((v.p_step + v.p_cell) / 2) and v.caveats == ["laser protocol differs"]
    assert s.cell_features()["len_par_base"] > 0


@pytest.mark.skipif(not MODEL.exists(), reason="run scripts/classification/save_final_framework.py")
def test_saved_framework_reproduces_the_training_cells():
    m = fw.load(MODEL)
    assert m.info["loocv_correct"] == 14 and len(m.info["train_cells"]) == 15
    res = ROOT / "results/features/nnunet1000_loocv"
    cells = pd.read_csv(res / "features_cells.csv")
    cells = cells[cells.usable]
    frames = pd.read_csv(res / "features_frames.csv")
    frames = frames[frames.passed & frames.cell.isin(cells.cell)]
    right = 0
    for _, c in cells.iterrows():
        st = frames[frames.cell == c.cell].groupby("laser_step")[fw.STEP_FEATURES].median().reset_index()
        p = m.p_high(st, c.to_dict())["p_high"]
        right += (p >= 0.5) == (c.her2 == "high")
    assert right >= 13           # fitted on these cells, so this checks the plumbing, not the accuracy


def test_framework_load_checks_what_it_opened(tmp_path):
    import joblib
    from celldform.demo.framework import HER2Framework, load, save
    save(fake_framework(), tmp_path / "ok.joblib")
    assert isinstance(load(tmp_path / "ok.joblib"), HER2Framework)
    joblib.dump({"a": 1}, tmp_path / "dict.joblib")
    with pytest.raises(ValueError, match="holds a dict, not a HER2 framework"):
        load(tmp_path / "dict.joblib")
    (tmp_path / "junk.joblib").write_bytes(b"junk")
    with pytest.raises(ValueError, match="could not be read"):
        load(tmp_path / "junk.joblib")
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "none.joblib")


# ── provisional verdicts: computed when validation fails, never mistaken for a result ───────────────────────────

def test_the_default_still_withholds_a_verdict_that_fails_validation():
    s = CellSession(fake_framework())
    feed(s, 400)                                                  # 2 laser steps: under the minimum of 3
    v = s.verdict()
    assert v.status == "insufficient" and v.label is None and not v.validated and "at least 3" in v.reason


def test_a_short_recording_gets_a_labelled_step_model_verdict_without_the_provisional_option():
    s = CellSession(fake_framework())                             # provisional is OFF
    feed(s, 4 * 200)                                              # 4 laser steps: more than 2, fewer than 10
    v = s.verdict()
    assert v.status == "short" and v.label in ("high", "low") and v.n_steps == 4 and not v.validated
    assert v.p_high == v.p_step and v.p_cell != v.p_cell                    # the step model alone; no cell model
    assert "step model" in v.reason and "10 laser steps" in v.reason and "size and shape" in v.reason
    assert v.headline().startswith("SHORT RECORDING, STEP MODEL ONLY: HER2 ") and f"{v.p_high:.2f}" in v.headline()


@pytest.mark.parametrize("steps, status", [(2, "insufficient"), (3, "short"), (9, "short"), (10, "ok")])
def test_the_step_count_boundaries(steps, status):
    s = CellSession(fake_framework())
    feed(s, steps * 200)
    v = s.verdict()
    assert v.n_steps == steps and v.status == status


def test_a_short_recording_that_fails_the_frame_checks_still_gets_no_verdict():
    s = CellSession(fake_framework())
    k = 0
    for i in range(800):
        if s.wants(i):
            s.add(seg(x=100 if k % 4 == 3 else 600), i)
            k += 1
    v = s.verdict()
    assert v.status == "insufficient" and "of measured frames pass the checks" in v.reason


def test_a_short_recording_with_a_failed_pass_rate_is_provisional_with_the_old_imputed_cell_model():
    s = CellSession(fake_framework(), provisional=True)
    k = 0
    for i in range(800):
        if s.wants(i):
            s.add(seg(x=100 if k % 4 == 3 else 600), i)
            k += 1
    v = s.verdict()
    assert v.status == "provisional" and v.p_cell == v.p_cell


def test_provisional_gives_a_verdict_when_there_are_too_few_steps_and_says_so():
    s = CellSession(fake_framework(), provisional=True)
    feed(s, 400)
    v = s.verdict()
    assert v.status == "provisional" and v.label in ("high", "low") and 0 <= v.p_high <= 1 and not v.validated
    assert v.p_high == pytest.approx((v.p_step + v.p_cell) / 2) and v.n_steps == 2
    assert "laser steps measured; at least 3 are needed" in v.reason and "validated only on cells that pass" in v.reason
    assert v.headline().startswith("PROVISIONAL, NOT VALIDATED: HER2 ") and f"{v.p_high:.2f}" in v.headline()


def test_provisional_with_a_low_pass_rate_and_both_reasons_listed():
    s = CellSession(fake_framework(), provisional=True)
    k = 0
    for i in range(400):                                          # few steps AND every 4th frame jumps away
        if s.wants(i):
            s.add(seg(x=100 if k % 4 == 3 else 600), i)
            k += 1
    v = s.verdict()
    assert v.status == "provisional"
    assert "laser steps measured" in v.reason and "of measured frames pass the checks" in v.reason


def test_provisional_notes_cell_measures_that_were_filled_in(monkeypatch):
    s = CellSession(fake_framework(), provisional=True)
    feed(s, 400)
    monkeypatch.setattr(s, "cell_features", lambda: {})           # nothing could be measured at cell level
    v = s.verdict()
    assert v.status == "provisional" and "4 of 4 cell-level measures could not be computed" in v.reason
    assert "median" in v.reason


def test_provisional_cannot_invent_a_verdict_without_a_single_usable_step():
    s = CellSession(fake_framework(), provisional=True)
    for i in range(400):
        if s.wants(i):
            s.add(seg(r=1), i)                                    # a speck: no frame passes
    v = s.verdict()
    assert v.status == "insufficient" and v.label is None and v.n_steps == 0


def test_provisional_without_a_classifier_is_still_no_verdict():
    s = CellSession(None, provisional=True)
    feed(s, 800)
    assert s.verdict().status == "insufficient" and "no classifier" in s.verdict().reason


def test_a_session_that_passes_validation_is_ok_even_in_provisional_mode():
    s = CellSession(fake_framework(), provisional=True)
    for i in range(2600):
        if s.wants(i):
            s.add(seg(r=16, ax=1.0 + 0.02 * (i // 200)), i)
    v = s.verdict()
    assert v.status == "ok" and v.validated and v.headline() == f"HER2 {v.label} (P high {v.p_high:.2f})"


def test_headlines_for_every_status():
    from celldform.demo.session import Verdict
    assert Verdict("insufficient", "why").headline() == "no verdict - why"
    assert Verdict("ok", "", "low", 0.2).headline() == "HER2 low (P high 0.20)"
    assert Verdict("short", "r", "low", 0.4).headline() == "SHORT RECORDING, STEP MODEL ONLY: HER2 low (P high 0.40)"
    assert Verdict("provisional", "r", "high", 0.7).headline() == "PROVISIONAL, NOT VALIDATED: HER2 high (P high 0.70)"

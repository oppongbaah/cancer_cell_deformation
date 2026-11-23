"""Tests for celldform.demo.dashboard: every panel draws with no data, partial data and a full verdict."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from celldform.demo import dashboard as D
from celldform.demo.session import CellSession
from tests.test_demo_session import fake_framework, feed


def _log(n):
    return type("Rec", (), {"frame_log": pd.DataFrame({"time_s": np.arange(n) / 22.0,
                                                       "set_current_mA": 350 + 650 * np.arange(n) / n}),
                            "declared_frames": n, "cell_label": "SKBR3", "path": "x"})()


def _session(n_frames):
    s = CellSession(fake_framework(), caveats=["c1"])
    feed(s, n_frames)
    return s


def _draw_all(state):
    fig = Figure(figsize=(9, 3.2))
    for page in D.PAGES:
        D.draw_page(fig, state, page)
        fig.canvas.draw() if fig.canvas else None


def test_reason_groups():
    assert D.reason_group("no cell near the trap") == "no cell near the trap"
    assert D.reason_group("93 px from the trap") == "too far from the trap"
    assert D.reason_group("boundary confidence 0.94") == "uncertain boundary"
    assert D.reason_group("area x1.40 of typical") == "size off typical"
    assert D.reason_group("something new") == "other"


def test_empty_state_draws_every_page():
    _draw_all(D.State())


def test_protocol_only_state_draws():
    n = 4400
    _draw_all(D.State(t_s=np.arange(n) / 22.0, mA=np.linspace(350, 1000, n), total_frames=n, title="x"))


def test_short_session_state_shows_what_is_missing():
    s = _session(400)
    st = D.build_state(s, _log(400), n_measured=30, elapsed_s=3.0, segmenter="unet", device="cpu")
    assert st.verdict.status == "insufficient" and len(st.steps) < st.min_steps
    assert len(st.frames) == 12 and st.mA is not None and len(st.crops) == 0   # no gray passed: no pictures
    _draw_all(st)


def test_short_recording_state_draws_the_flagged_gauge_without_a_cell_model():
    s = _session(1000)                                                          # 5 laser steps
    st = D.build_state(s, _log(1000), n_measured=30, elapsed_s=3.0, segmenter="unet", device="cpu")
    assert st.verdict.status == "short" and st.verdict.p_cell != st.verdict.p_cell
    _draw_all(st)
    fig = Figure(figsize=(9, 3.2)); ax = fig.subplots(1, 1)
    D.panel_verdict(ax, st)
    text = " ".join(t.get_text() for t in ax.texts)
    assert "SHORT RECORDING: STEP MODEL ONLY" in text and "cell model not used" in text and "nan" not in text


def test_full_session_state_draws_the_gauge_and_reference():
    s = _session(2400)
    ref = pd.DataFrame({"her2": ["high", "low"] * 4, "len_par_base": np.linspace(28, 42, 8),
                        "fluctuation": np.linspace(0.003, 0.03, 8)})
    st = D.build_state(s, _log(2400), n_measured=len(s.frames), elapsed_s=5.0, reference=ref)
    assert st.verdict.status == "ok"
    fig = Figure()
    ax = fig.subplots()
    D.panel_verdict(ax, st)
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "HER2" in texts and "P(high)" in texts and "step model" in texts
    _draw_all(st)


def test_missing_reference_and_log_do_not_crash():
    st = D.build_state(_session(1000), None, n_measured=5)
    st.reference = None
    assert st.mA is None
    _draw_all(st)


def test_save_writes_a_png(tmp_path):
    s = _session(2400)
    st = D.build_state(s, _log(2400), n_measured=len(s.frames), elapsed_s=5.0)
    out = D.save(st, tmp_path / "d.png", dpi=60)
    assert out.is_file() and out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_make_crop_follows_the_region_and_handles_none():
    gray = np.full((300, 400), 120, np.uint8)
    region = np.zeros((300, 400), bool); region[140:160, 300:320] = True
    c = D.make_crop(gray, region)
    assert c.shape == (D.CROP, D.CROP, 3)
    assert (c > 150).any()                                     # the white outline is drawn
    assert D.make_crop(gray, None) is None and D.make_crop(gray, np.zeros_like(region)) is None


def test_session_keeps_one_picture_per_step_when_given_the_frame():
    s = CellSession(fake_framework())
    from tests.test_demo_session import seg
    gray = np.full((922, 1228), 100, np.uint8)
    for i in range(0, 600):
        if s.wants(i):
            s.add(seg(), i, gray)
    assert sorted(s.crops) == [0, 1, 2] and all(c.ndim == 3 for c in s.crops.values())


def test_a_provisional_verdict_is_labelled_on_the_dashboard(tmp_path):
    from celldform.demo.session import Verdict
    v = Verdict("provisional", "only 13% of measured frames pass the checks (need 80%). The classifier was validated "
                "only on cells that pass these checks", "high", 0.52, 0.54, 0.51, 12, 0.13)
    st = D.State(verdict=v, steps=pd.DataFrame({"a": range(12)}))
    ax = Figure().subplots()
    D.panel_verdict(ax, st)
    text = " ".join(t.get_text() for t in ax.texts)
    assert "HIGH HER2" in text and "PROVISIONAL: NOT VALIDATED" in text and "Not validated: only 13%" in text
    assert "step model 0.54" in text and "average 0.52" in text
    ok = Verdict("ok", "", "high", 0.9, 0.9, 0.9, 12, 0.9)
    ax2 = Figure().subplots()
    D.panel_verdict(ax2, D.State(verdict=ok, steps=pd.DataFrame({"a": range(12)})))
    assert "PROVISIONAL" not in " ".join(t.get_text() for t in ax2.texts)          # a validated one carries no such label
    out = D.save(st, tmp_path / "p.png", dpi=50)
    assert out.is_file()

"""Tests for celldform.demo.gui (stage 8), offscreen."""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("celldform.demo.qt")
from celldform.demo.qt import QtWidgets

from celldform.demo.gui import MainWindow, map_click, to_qimage
from tests.test_demo_run import DiscSegmenter, make_recording
from tests.test_demo_session import fake_framework


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_map_click_centres_and_rejects_the_bars():
    # 1228x922 image in a 1228x1000 label: scale 1, 39 px bars top and bottom
    assert map_click((100, 39 + 50), (1228, 1000)) == pytest.approx((100, 50))
    assert map_click((100, 10), (1228, 1000)) is None
    assert map_click((614, 500), (614, 461)) is None            # outside on the right? x within, y beyond
    x, y = map_click((307, 230), (614, 461))                     # half scale
    assert (x, y) == pytest.approx((614, 460), abs=3)


def test_to_qimage_swaps_bgr_to_rgb():
    img = np.zeros((4, 5, 3), np.uint8); img[..., 0] = 255      # pure blue in BGR
    q = to_qimage(img)
    assert (q.width(), q.height()) == (5, 4) and q.pixelColor(0, 0).blue() == 255 and q.pixelColor(0, 0).red() == 0


def wait(cond, secs=30):
    t = time.time()
    while not cond() and time.time() - t < secs:
        QtWidgets.QApplication.processEvents(); time.sleep(0.01)
    QtWidgets.QApplication.processEvents()
    return cond()


def test_window_runs_a_recording_and_shows_the_verdict(app, tmp_path, monkeypatch):
    import celldform.demo.source as src
    monkeypatch.setattr(src, "FEATURE_SIZE", (160, 120))
    import celldform.demo.run as runmod
    orig = runmod.run
    monkeypatch.setattr("celldform.demo.gui.run", lambda *a, **k: orig(*a, size=(160, 120), **k))
    rec = make_recording(tmp_path / "r", 450)
    w = MainWindow(DiscSegmenter(), fake_framework())
    assert not w.btn_analyze.isEnabled()
    w.load(str(rec))
    assert w.btn_analyze.isEnabled() and "450 frames" in w.info.text()
    w.on_click(10, 10)
    assert w.trap is None                                       # clicks are ignored until "Click the cell"
    w.pick_trap(); w.on_click(80, 60)
    assert w.trap == pytest.approx((614, 461), abs=1)         # the middle of the 160x120 frame, in 1228x922 analysis pixels
    w.start()
    assert wait(lambda: w.worker.isFinished())
    assert w.worker.verdict is not None and w.worker.verdict.status == "insufficient"
    assert w.verdict_lbl.text().startswith("Finished: no HER2 verdict") and "laser steps" in w.info.text()
    assert w.btn_analyze.isEnabled() and not w.btn_stop.isEnabled()
    assert w.dash.state.n_measured == 14 and w.dash.state.verdict is not None      # the dashboard got the final snapshot
    assert (w.out / "dashboard.png").is_file()


def test_missing_recording_shows_error_not_crash(app, tmp_path):
    w = MainWindow(DiscSegmenter(), None)
    w.load(str(tmp_path))
    assert w.verdict_lbl.text().startswith("Cannot open") and not w.btn_analyze.isEnabled()


def test_dashboard_tabs_redraw_on_every_page(app):
    from celldform.demo.gui import DashboardTabs
    t = DashboardTabs()
    assert [t.tabText(i) for i in range(t.count())] == ["Process", "Evidence", "Result and model", "Log"]
    for i in range(t.count()):
        t.setCurrentIndex(i)
        t.redraw()


class SlowSegmenter(DiscSegmenter):
    def segment(self, gray):
        time.sleep(0.04)
        return super().segment(gray)


def _patched_run(monkeypatch):
    import celldform.demo.source as src
    monkeypatch.setattr(src, "FEATURE_SIZE", (160, 120))
    import celldform.demo.run as runmod
    orig = runmod.run
    monkeypatch.setattr("celldform.demo.gui.run", lambda *a, **k: orig(*a, size=(160, 120), **k))


def test_buttons_are_real_buttons_with_the_right_states(app, tmp_path):
    w = MainWindow(DiscSegmenter(), fake_framework())
    assert [b.text() for b in (w.btn_analyze, w.btn_pause, w.btn_stop)] == ["Analyze", "Pause", "Stop"]
    assert not any(b.isEnabled() for b in (w.btn_analyze, w.btn_pause, w.btn_stop))
    w.load(str(make_recording(tmp_path / "r", 450)))
    assert w.btn_analyze.isEnabled() and not w.btn_pause.isEnabled() and not w.btn_stop.isEnabled()


def test_pause_holds_the_run_and_resume_finishes_it(app, tmp_path, monkeypatch):
    _patched_run(monkeypatch)
    w = MainWindow(SlowSegmenter(), fake_framework())
    w.load(str(make_recording(tmp_path / "r", 1500)))
    w.btn_analyze.click()
    assert w.btn_analyze.isEnabled() is False and w.btn_pause.isEnabled() and w.btn_stop.isEnabled()
    assert wait(lambda: w.dash.log.toPlainText().count(" PASS") + w.dash.log.toPlainText().count(" REJECT") >= 3)
    w.btn_pause.click()
    assert w.btn_pause.text() == "Resume" and w.worker.paused
    assert wait(lambda: "paused" in w.dash.log.toPlainText().splitlines()[-1])
    n = len(w.dash.log.toPlainText().splitlines())
    wait(lambda: False, 0.4)
    assert len(w.dash.log.toPlainText().splitlines()) == n and w.worker.isRunning()      # nothing moves while paused
    w.btn_pause.click()
    assert w.btn_pause.text() == "Pause"
    assert wait(lambda: w.worker.isFinished(), 30)
    log = w.dash.log.toPlainText()
    assert "resumed after" in log and "finished:" in log
    assert not w.btn_pause.isEnabled() and w.btn_analyze.isEnabled()
    import json
    j = json.loads((w.out / "verdict.json").read_text())
    assert j["paused_seconds"] >= 0.3 and j["seconds"] >= 0                     # pauses are not counted as speed


def test_stop_works_while_paused(app, tmp_path, monkeypatch):
    _patched_run(monkeypatch)
    w = MainWindow(SlowSegmenter(), fake_framework())
    w.load(str(make_recording(tmp_path / "r", 1500)))
    w.btn_analyze.click()
    assert wait(lambda: " PASS" in w.dash.log.toPlainText() or " REJECT" in w.dash.log.toPlainText())
    w.btn_pause.click()
    assert wait(lambda: "paused" in w.dash.log.toPlainText().splitlines()[-1])
    w.btn_stop.click()
    assert wait(lambda: w.worker.isFinished(), 10)
    assert "stopped:" in w.dash.log.toPlainText() and w.statusBar().currentMessage().startswith("Stopped")


def test_log_tab_shows_the_inference_log_and_start_clears_it(app, tmp_path, monkeypatch):
    _patched_run(monkeypatch)
    w = MainWindow(DiscSegmenter(), fake_framework())
    w.load(str(make_recording(tmp_path / "r", 450)))
    log_tab = [w.dash.tabText(i) for i in range(w.dash.count())].index("Log")
    w.btn_analyze.click()
    assert wait(lambda: w.worker.isFinished())
    text = w.dash.log.toPlainText()
    assert "start:" in text and "result: no verdict" in text and text.count("seg ") == 14
    assert (w.out / "inference.log").read_text().strip() == text.strip()
    w.dash.setCurrentIndex(log_tab); w.dash.redraw()                           # drawing the text tab must not fail
    w.btn_analyze.click()
    assert wait(lambda: w.worker.isFinished())
    assert w.dash.log.toPlainText().count("start:") == 1                       # cleared, not appended to


def test_source_box_switches_between_video_and_images(app, tmp_path, monkeypatch):
    import cv2
    import pandas as pd
    folder = make_recording(tmp_path / "r", 450)
    w = MainWindow(DiscSegmenter(), fake_framework())
    w.load(str(folder))
    assert [w.source_box.itemData(i) for i in range(2)] == ["video", "images"]
    w.source_box.setCurrentIndex(1)                                            # no stills yet: a clear error ...
    assert w.verdict_lbl.text().startswith("Cannot open")
    assert w.source_box.currentData() == "video" and w.btn_analyze.isEnabled()   # ... and the video stays usable
    d = folder / "stills_1GB"; d.mkdir()
    names = []
    for k in range(450):
        names.append(f"s{k:04d}.png")
        cv2.imwrite(str(d / names[-1]), np.full((120, 160, 3), 120, np.uint8))
    pd.DataFrame({"still_file": names, "video_frame": range(1, 451)}).to_csv(folder / "stills.csv", index=False)
    w.source_box.setCurrentIndex(0); w.source_box.setCurrentIndex(1)
    assert "450 still images" in w.info.text() and w.btn_analyze.isEnabled()
    _patched_run(monkeypatch)
    w.btn_analyze.click()
    assert not w.source_box.isEnabled()                                        # locked while running
    assert wait(lambda: w.worker.isFinished())
    assert "source=images" in w.dash.log.toPlainText() and w.source_box.isEnabled()
    assert w.dash.state.source == "images"


# ── Pause must hold the run at once, even when the window is slow ───────────────────────────────────────────────

def test_latest_hands_over_only_the_newest_and_says_so_once():
    from celldform.demo.gui import Latest
    q = Latest()
    assert q.put("a") is True                      # the window has to be told
    assert q.put("b") is False and q.put("c") is False        # it is already on its way: no queue builds up
    assert q.take() == "c" and q.take() is None
    assert q.put("d") is True                      # after it collected, the next item tells it again


def test_log_buffer_never_drops_a_line():
    from celldform.demo.gui import LogBuffer
    b = LogBuffer()
    assert b.put("1") and not b.put("2") and not b.put("3")
    assert b.take() == ["1", "2", "3"] and b.take() == []
    assert b.put("4")


def test_the_runner_offers_a_tick_for_every_frame_it_steps_over(tmp_path):
    from celldform.demo.run import run
    calls = []
    run(make_recording(tmp_path / "r", 450), tmp_path / "out", DiscSegmenter(), fake_framework(), size=(160, 120),
        tick=lambda: calls.append(1), write_video=False)
    assert 450 <= len(calls) <= 451                # every frame (plus the look past the end), not only the 14 measured


def test_pause_stops_the_video_at_once_even_if_the_window_is_slow(app, tmp_path, monkeypatch):
    """Regression: with a window slower than the worker, frames queued up and kept playing for seconds after Pause."""
    _patched_run(monkeypatch)
    w = MainWindow(SlowSegmenter(), fake_framework())
    w.load(str(make_recording(tmp_path / "r", 3000)))
    shown = []
    orig = w.on_frame

    def slow_window(img, v, i=0):                  # a window that needs 0.12 s per frame; the worker makes one every 0.04
        time.sleep(0.12)
        shown.append(time.perf_counter())
        orig(img, v, i)
    monkeypatch.setattr(w, "on_frame", slow_window)
    w.btn_analyze.click()
    assert wait(lambda: len(shown) >= 4)
    t = time.perf_counter()
    w.btn_pause.click()
    assert w.view.banner == "PAUSED" and w.state_lbl.text() == "Paused" and w.btn_pause.text() == "Resume"
    wait(lambda: False, 0.7)
    after = [s for s in shown if s > t]
    assert len(after) <= 2, f"{len(after)} frames still shown after Pause"
    n = len(shown)
    wait(lambda: False, 0.5)
    assert len(shown) == n                         # and then nothing moves
    assert w.worker.isRunning()
    w.btn_pause.click()                            # resume
    assert w.view.banner == "" and w.state_lbl.text() == "Analyzing"
    assert wait(lambda: len(shown) > n)            # frames flow again
    w.btn_stop.click()
    assert wait(lambda: w.worker.isFinished(), 20)
    assert w.state_lbl.text() == "Idle"


def test_pause_pressed_right_after_start_holds_before_the_first_frame(app, tmp_path, monkeypatch):
    _patched_run(monkeypatch)
    w = MainWindow(SlowSegmenter(), fake_framework())
    w.load(str(make_recording(tmp_path / "r", 450)))
    w.btn_analyze.click()
    w.btn_pause.click()                            # before the worker has read anything
    assert wait(lambda: "paused" in w.dash.log.toPlainText(), 10)
    wait(lambda: False, 0.4)
    assert " PASS" not in w.dash.log.toPlainText() and " REJECT" not in w.dash.log.toPlainText()
    w.btn_pause.click()
    assert wait(lambda: w.worker.isFinished(), 20)
    assert w.dash.log.toPlainText().count("seg ") == 14


def test_the_dashboard_and_log_still_get_everything_when_the_window_is_slow(app, tmp_path, monkeypatch):
    _patched_run(monkeypatch)
    w = MainWindow(DiscSegmenter(), fake_framework())
    w.load(str(make_recording(tmp_path / "r", 1000)))
    orig = w.dash.set_state
    monkeypatch.setattr(w.dash, "set_state", lambda s: (time.sleep(0.2), orig(s)))
    w.btn_analyze.click()
    assert wait(lambda: w.worker.isFinished(), 30)
    assert w.dash.log.toPlainText().count("seg ") == 30                # 5 steps x 6: no log line lost
    assert w.dash.state.n_measured == 30 and len(w.dash.state.frames) == 30        # the final snapshot is complete


# ── the provisional-verdict option ──────────────────────────────────────────────────────────────────────────────

def test_provisional_is_off_by_default_and_explains_itself(app):
    w = MainWindow()
    assert not w.chk_provisional.isChecked()
    tip = w.chk_provisional.toolTip()
    assert "Off by default" in tip and "NOT VALIDATED" in tip and "not a HER2 result" in tip


def test_without_the_option_a_failed_validation_gives_no_verdict(app, tmp_path, monkeypatch):
    _patched_run(monkeypatch)
    w = MainWindow(DiscSegmenter(), fake_framework())
    w.load(str(make_recording(tmp_path / "r", 450)))
    w.btn_analyze.click()
    assert wait(lambda: w.worker.isFinished())
    assert w.verdict_lbl.text().startswith("Finished: no HER2 verdict") and "PROVISIONAL" not in w.verdict_lbl.text()


def test_with_the_option_the_verdict_is_shown_as_provisional_everywhere(app, tmp_path, monkeypatch):
    import json
    _patched_run(monkeypatch)
    w = MainWindow(DiscSegmenter(), fake_framework())
    w.load(str(make_recording(tmp_path / "r", 450)))
    w.chk_provisional.setChecked(True)
    w.btn_analyze.click()
    assert not w.chk_provisional.isEnabled()                                # locked while running
    assert wait(lambda: w.worker.isFinished())
    assert w.chk_provisional.isEnabled()
    assert w.verdict_lbl.text().startswith("PROVISIONAL, not validated:") and "HER2" in w.verdict_lbl.text()
    assert "color:#b85c00" in w.verdict_lbl.styleSheet()
    assert w.info.text().startswith("Not a HER2 result, because:") and "needs 10" in w.info.text()
    log = w.dash.log.toPlainText()
    assert "result: PROVISIONAL, NOT VALIDATED" in log
    j = json.loads((w.out / "verdict.json").read_text())
    assert j["status"] == "provisional" and j["validated"] is False
    # a normal verdict afterwards is not left orange
    w.chk_provisional.setChecked(False)
    w.btn_analyze.click()
    assert wait(lambda: w.worker.isFinished())
    assert "PROVISIONAL" not in w.verdict_lbl.text() and "b85c00" not in w.verdict_lbl.styleSheet()

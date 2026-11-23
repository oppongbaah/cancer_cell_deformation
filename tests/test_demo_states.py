"""Every stage of the window: what can be pressed, and that Analyze / Pause / Stop work from each one. Offscreen."""
import os
import random
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("celldform.demo.qt")
from celldform.demo.qt import QtWidgets

from celldform.demo.gui import MainWindow, fix_qt_plugin_path
from celldform.demo.segment import Seg
from tests.test_demo_browse import make_rec
from tests.test_demo_gui import SlowSegmenter, _patched_run, make_recording
from tests.test_demo_run import DiscSegmenter
from tests.test_demo_session import fake_framework


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait(cond, secs=15.0):
    t = time.time()
    while not cond() and time.time() - t < secs:
        QtWidgets.QApplication.processEvents(); time.sleep(0.005)
    QtWidgets.QApplication.processEvents()
    return cond()


def settle(w, secs=20.0):
    assert wait(lambda: w._run_state == "idle", secs), f"stuck in {w._run_state}"


def give_models(w):
    """What the window has when the required files are loaded (here: ready-made fakes)."""
    from celldform.demo import models as M
    w.segmenter, w.framework = SlowSegmenter(), fake_framework()
    w.model_checks.update({r: M.Check(r, None, True, "test") for r in M.REQUIRED})
    w._refresh_models_ui()


def window(tmp_path, monkeypatch, n=900, segmenter=None):
    _patched_run(monkeypatch)
    tmp_path.mkdir(parents=True, exist_ok=True)
    w = MainWindow(segmenter or SlowSegmenter(), fake_framework())
    w.load(str(make_recording(tmp_path / "rec", n)))
    return w


def invariants(w):
    """What the buttons must say, whatever happened before. The single source of truth is (loaded, run state)."""
    st = w._run_state
    idle, active = st == "idle", st in ("running", "paused")
    assert w.btn_analyze.isEnabled() == (w.loaded and idle and w.models_ready), f"Analyze wrong in {st}"
    assert w.btn_pause.isEnabled() == active and w.btn_stop.isEnabled() == active, f"Pause/Stop wrong in {st}"
    assert w.btn_pause.text() == ("Resume" if st == "paused" else "Pause")
    assert w.player.isEnabled() == (w.loaded and idle), f"Browse bar wrong in {st}"
    assert w.open_btn.isEnabled() == idle and w.models_box.isEnabled() == idle and w.chk_provisional.isEnabled() == idle
    assert w.source_box.isEnabled() == (idle and not w._images_only)
    assert (w.view.banner == "PAUSED") == (st == "paused")
    assert w.state_lbl.text() == {"idle": "Idle", "running": "Analyzing", "paused": "Paused", "stopping": "Stopping"}[st]
    if st != "idle":
        assert not w.player.playing


# ── each stage ──────────────────────────────────────────────────────────────────────────────────────────────────

def test_nothing_loaded_nothing_can_be_pressed_and_analyze_says_why(app):
    w = MainWindow()
    invariants(w)
    assert not any(b.isEnabled() for b in (w.btn_analyze, w.btn_pause, w.btn_stop)) and w.btn_analyze.text() == "Analyze"
    w.btn_analyze.click(); w.btn_pause.click(); w.btn_stop.click()           # disabled: nothing happens
    w.start()                                                              # called anyway: explained, not silent
    assert "Open a recording first" in w.statusBar().currentMessage() and w._run_state == "idle"
    w.toggle_pause(); w.stop()
    assert w._run_state == "idle"
    invariants(w)


def test_loaded_and_idle(app, tmp_path, monkeypatch):
    w = window(tmp_path, monkeypatch)
    invariants(w)
    assert w.btn_analyze.isEnabled() and w.player.isEnabled()


def test_analyze_after_playing_the_video_in_every_playback_state(app, tmp_path, monkeypatch):
    """The reported problem: browse-play, then Analyze."""
    for variant in ("while playing", "after pausing the playback", "after playing to the end", "after scrubbing and stepping"):
        w = window(tmp_path / variant.replace(" ", "_"), monkeypatch)
        if variant == "while playing":
            w.player.btn_play.click(); wait(lambda: w.player.idx > 3)
            assert w.player.playing
        elif variant == "after pausing the playback":
            w.player.btn_play.click(); wait(lambda: w.player.idx > 3); w.player.btn_play.click()
            assert not w.player.playing
        elif variant == "after playing to the end":
            w.player.speed.setCurrentIndex(6); w.player.show_frame(w.player.count - 30); w.player.btn_play.click()
            wait(lambda: not w.player.playing, 10)
            assert w.player.idx == w.player.count - 1
        else:
            w.player.slider.setValue(300); w.player.step(1); w.player.step(-1); w.player.btn_last.click(); w.player.btn_first.click()
        invariants(w)
        w.btn_analyze.click()
        assert w._run_state == "running", variant
        invariants(w)
        assert wait(lambda: "seg " in w.dash.log.toPlainText(), 15), f"no analysis output {variant}"   # it really analyzes
        assert not w.player.playing and not w.player.isEnabled()
        w.btn_stop.click(); settle(w)
        invariants(w)


def test_analyze_pause_resume_finish_then_analyze_again(app, tmp_path, monkeypatch):
    w = window(tmp_path, monkeypatch, n=1200)
    w.btn_analyze.click(); invariants(w)
    assert wait(lambda: w.dash.log.toPlainText().count("seg ") >= 3)
    w.btn_pause.click(); invariants(w)
    assert w._run_state == "paused" and w.btn_pause.text() == "Resume"
    n = w.dash.log.toPlainText().count("seg ")
    wait(lambda: False, 0.5)
    assert w.dash.log.toPlainText().count("seg ") <= n + 1                 # held (one frame may be in flight)
    w.btn_pause.click(); invariants(w)
    assert w._run_state == "running"
    settle(w)
    invariants(w)
    first = w.dash.log.toPlainText()
    assert "finished:" in first
    w.btn_analyze.click(); invariants(w)                                    # second run, straight after the first
    assert wait(lambda: "seg " in w.dash.log.toPlainText())
    assert w.dash.log.toPlainText().count("start:") == 1                    # fresh log
    settle(w)
    assert "finished:" in w.dash.log.toPlainText()


def test_pause_then_stop_does_not_stay_stuck(app, tmp_path, monkeypatch):
    w = window(tmp_path, monkeypatch, n=1200)
    w.btn_analyze.click()
    assert wait(lambda: "seg " in w.dash.log.toPlainText())
    w.btn_pause.click()
    w.btn_stop.click(); invariants(w)
    assert w._run_state == "stopping" and not w.btn_analyze.isEnabled()
    settle(w); invariants(w)
    assert w.btn_analyze.isEnabled() and w.statusBar().currentMessage().startswith("Stopped")
    w.btn_analyze.click()
    assert wait(lambda: w.dash.log.toPlainText().count("seg ") >= 2)       # and it analyzes again
    w.btn_stop.click(); settle(w)


def test_analyze_again_right_after_stop(app, tmp_path, monkeypatch):
    w = window(tmp_path, monkeypatch, n=1200)
    w.btn_analyze.click()
    assert wait(lambda: "seg " in w.dash.log.toPlainText())
    w.btn_stop.click()
    w.btn_analyze.click()                                                   # still stopping: ignored, not queued or lost
    assert w._run_state == "stopping"
    settle(w)
    w.btn_analyze.click()
    assert w._run_state == "running"
    assert wait(lambda: "seg " in w.dash.log.toPlainText())
    w.btn_stop.click(); settle(w)


def test_pause_resume_pause_resume_in_quick_succession(app, tmp_path, monkeypatch):
    w = window(tmp_path, monkeypatch, n=1500)
    w.btn_analyze.click()
    for _ in range(6):
        w.btn_pause.click(); invariants(w)
        w.btn_pause.click(); invariants(w)
    assert w._run_state == "running"
    assert wait(lambda: w.dash.log.toPlainText().count("seg ") >= 3)
    w.btn_stop.click(); settle(w)


def test_pause_before_the_first_frame_then_stop(app, tmp_path, monkeypatch):
    w = window(tmp_path, monkeypatch)
    w.btn_analyze.click(); w.btn_pause.click(); invariants(w)
    assert wait(lambda: "paused" in w.dash.log.toPlainText())
    w.btn_stop.click(); settle(w); invariants(w)
    assert w.dash.log.toPlainText().count("seg ") <= 2                     # at most the frame already in flight


def test_double_click_on_analyze_starts_one_run(app, tmp_path, monkeypatch):
    w = window(tmp_path, monkeypatch, n=1200)
    w.btn_analyze.click(); w.btn_analyze.click(); w.start()
    assert w._run_state == "running" and "already running" in w.statusBar().currentMessage()
    assert wait(lambda: "seg " in w.dash.log.toPlainText())
    assert w.dash.log.toPlainText().count("start:") == 1
    w.btn_stop.click(); settle(w)


def test_nothing_else_can_be_opened_or_changed_while_analyzing(app, tmp_path, monkeypatch):
    w = window(tmp_path, monkeypatch, n=1200)
    other = make_rec(tmp_path / "other")
    w.btn_analyze.click()
    before = w.recording
    w.open_paths([str(other)])
    assert w.recording == before and "Stop the analysis" in w.statusBar().currentMessage()
    w.load(str(other))
    assert w.recording == before
    assert not w.open_btn.isEnabled() and not w.source_box.isEnabled() and not w.models_box.isEnabled()
    w.btn_pause.click()
    w.open_paths([str(other)]); assert w.recording == before                 # also while paused
    w.btn_stop.click(); settle(w)
    w.open_paths([str(other)])
    assert w.recording != before                                             # open again once idle
    invariants(w)


def test_a_failed_open_keeps_the_loaded_recording_usable(app, tmp_path, monkeypatch):
    w = window(tmp_path, monkeypatch)
    (tmp_path / "junk.txt").write_text("x")
    for bad in ([str(tmp_path / "junk.txt")], [str(tmp_path / "missing.avi")], [str(tmp_path)]):
        w.open_paths(bad)
        assert "Cannot open" in w.statusBar().currentMessage()
        invariants(w)
        assert w.btn_analyze.isEnabled() and w.player.isEnabled()
    w.source_box.setCurrentIndex(1)                                          # no stills for this recording
    assert w.source_box.currentData() == "video" and w.btn_analyze.isEnabled()
    invariants(w)
    w.btn_analyze.click(); assert w._run_state == "running"
    w.btn_stop.click(); settle(w)


def test_an_error_inside_the_analysis_returns_to_idle_and_analyze_works_again(app, tmp_path, monkeypatch):
    class Boom:
        def segment(self, gray):
            raise RuntimeError("model exploded")
    w = window(tmp_path, monkeypatch, segmenter=Boom())
    w.btn_analyze.click()
    settle(w); invariants(w)
    assert "model exploded" in w.verdict_lbl.text() and "ERROR" in w.dash.log.toPlainText()
    assert w.btn_analyze.isEnabled()
    w.segmenter = DiscSegmenter()                                            # fixed: it runs
    w.btn_analyze.click()
    settle(w)
    assert "finished:" in w.dash.log.toPlainText() and w.verdict_lbl.text().startswith("Finished")


def test_an_error_while_starting_does_not_leave_the_window_stuck(app, tmp_path, monkeypatch):
    w = window(tmp_path, monkeypatch)
    import celldform.demo.gui as g
    monkeypatch.setattr(g, "Worker", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cannot start")))
    w.btn_analyze.click()
    assert w._run_state == "idle" and w.btn_analyze.isEnabled()
    assert "Analyze: RuntimeError: cannot start" in w.statusBar().currentMessage()
    assert "ERROR" in w.dash.log.toPlainText()
    invariants(w)


def test_closing_the_window_while_paused_ends_the_worker(app, tmp_path, monkeypatch):
    w = window(tmp_path, monkeypatch, n=1500)
    w.btn_analyze.click()
    assert wait(lambda: "seg " in w.dash.log.toPlainText())
    w.btn_pause.click()
    t = time.time()
    w.close()
    assert time.time() - t < 6 and not w.worker.isRunning()


def test_hand_picked_images_go_through_the_same_stages(app, tmp_path, monkeypatch):
    _patched_run(monkeypatch)
    folder = make_rec(tmp_path / "r", n=450)
    pics = [str(p) for p in sorted((folder / "stills_1GB").glob("*.png"))]
    w = MainWindow(SlowSegmenter(), fake_framework())
    w.open_paths(pics)
    invariants(w)
    w.player.btn_play.click(); wait(lambda: w.player.idx > 2)
    w.btn_analyze.click(); invariants(w)
    assert wait(lambda: "seg " in w.dash.log.toPlainText())
    w.btn_pause.click(); invariants(w); w.btn_pause.click(); invariants(w)
    settle(w); invariants(w)
    assert w.source_box.currentData() == "images" and not w.source_box.isEnabled()


# ── the OpenCV / Qt plugin clash and unexpected errors ──────────────────────────────────────────────────────────

def test_the_opencv_qt_plugin_path_is_undone(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM_PLUGIN_PATH", "/venv/lib/python3.12/site-packages/cv2/qt/plugins")
    monkeypatch.setenv("QT_QPA_FONTDIR", "/venv/lib/python3.12/site-packages/cv2/qt/fonts")
    assert fix_qt_plugin_path() is True
    assert "QT_QPA_PLATFORM_PLUGIN_PATH" not in os.environ and "QT_QPA_FONTDIR" not in os.environ
    monkeypatch.setenv("QT_QPA_PLATFORM_PLUGIN_PATH", "/opt/qt/plugins")
    assert fix_qt_plugin_path() is False and os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] == "/opt/qt/plugins"
    monkeypatch.delenv("QT_QPA_PLATFORM_PLUGIN_PATH")
    assert fix_qt_plugin_path() is False


def test_report_error_shows_and_logs_without_raising(app):
    w = MainWindow()
    w.report_error("Somewhere", ValueError("bad thing"))
    assert w.statusBar().currentMessage() == "Somewhere: ValueError: bad thing"
    assert "ERROR  Somewhere: ValueError: bad thing" in w.dash.log.toPlainText()


# ── random use ──────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_button_mashing_never_leaves_a_stage_broken(app, tmp_path, monkeypatch, seed):
    """Press things in random order (also the disabled ones, also while a run is going) and check, after every press, that
    the buttons say exactly what the stage allows. At the end the window must still analyze."""
    rnd = random.Random(seed)
    _patched_run(monkeypatch)
    w = MainWindow(SlowSegmenter(), fake_framework())
    w.load(str(make_recording(tmp_path / "rec", 900)))
    other = make_rec(tmp_path / "other")
    actions = {
        "analyze": lambda: w.btn_analyze.click(), "pause": lambda: w.btn_pause.click(), "stop": lambda: w.btn_stop.click(),
        "play": lambda: w.player.btn_play.click(), "step": lambda: w.player.step(1), "back": lambda: w.player.step(-1),
        "scrub": lambda: w.player.slider.setValue(rnd.randrange(w.player.slider.maximum() + 1)),
        "first": lambda: w.player.btn_first.click(), "last": lambda: w.player.btn_last.click(),
        "open_other": lambda: w.open_paths([str(other)]), "open_bad": lambda: w.open_paths([str(tmp_path / "nope.avi")]),
        "source": lambda: w.source_box.setCurrentIndex(1 - w.source_box.currentIndex()),
        "zoom": lambda: w.view._set_zoom(rnd.choice([1.0, 2.0, 5.0]), (100, 100)), "start_direct": lambda: w.start(),
        "clear_models": lambda: (w.clear_models(), give_models(w)),
    }
    names = list(actions)
    weights = [6, 5, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1]
    for _ in range(45):
        name = rnd.choices(names, weights)[0]
        actions[name]()
        wait(lambda: False, rnd.choice([0.0, 0.02, 0.1, 0.3]))
        invariants(w)
    if w._run_state in ("running", "paused"):
        w.btn_stop.click()
    settle(w)
    invariants(w)
    if w.source_box.currentData() != "video":
        w.source_box.setCurrentIndex(0)
    w.btn_analyze.click()
    assert w._run_state == "running"
    assert wait(lambda: "seg " in w.dash.log.toPlainText(), 20)
    w.btn_stop.click(); settle(w)

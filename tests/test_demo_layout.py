"""The window fills whatever size it has: stacked layout when small or tall, wide layout when maximized."""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("celldform.demo.qt")
from celldform.demo.qt import QtWidgets

from celldform.demo.gui import MainWindow, initial_size
from tests.test_demo_browse import make_rec
from tests.test_demo_run import DiscSegmenter
from tests.test_demo_session import fake_framework


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def pump(secs=0.4):
    t = time.time()
    while time.time() - t < secs:
        QtWidgets.QApplication.processEvents(); time.sleep(0.01)


def shown(w, size):
    w.resize(*size); w.show(); pump()
    return w


def test_wants_wide_has_a_margin_so_it_does_not_flip_at_the_border():
    f = MainWindow.wants_wide
    assert f(1914, 1047, False) and f(1920, 1080, False)           # a maximized 1080p screen
    assert not f(1366, 768, False)                                 # a small laptop screen stays stacked
    assert not f(1500, 1100, False)                                # too square to need it
    assert not f(1400, 900, False) and f(1400, 900, True)          # entering needs more than staying
    assert not f(1000, 900, True)


def test_the_initial_size_fits_the_screen():
    assert initial_size(1920, 1050) == (1700, 945)                 # was 1350x1120: taller than a 1080p screen
    w, h = initial_size(1366, 728)
    assert w <= 1366 and h <= 728
    assert initial_size(3840, 2100) == (1700, 1100)                # not absurdly large on a 4K screen


def test_a_big_window_uses_the_wide_layout_and_the_picture_fills_the_height(app, tmp_path):
    w = MainWindow()
    w.open_paths([str(make_rec(tmp_path / "r"))])
    shown(w, (1914, 1047))
    assert w._wide and w.pages.currentWidget() is w.page_wide
    assert w.view.height() > 800 and w.view.width() > 1000          # the picture, not black bars
    assert w.dash.parent() is w._slots["dash_b"] and w.side_scroll.parent() is not None
    assert w.dash.width() > 500 and w.dash.height() > 400             # the dashboard is on the right, readable
    assert w.centralWidget().width() >= 1900                          # nothing is left unused on the right


def test_a_small_window_uses_the_stacked_layout(app, tmp_path):
    w = MainWindow()
    shown(w, (1100, 800))
    assert not w._wide and w.pages.currentWidget() is w.page_stacked
    assert w.side_scroll.width() == 360
    assert w.dash.parent() is w._slots["dash_a"]


def test_resizing_back_and_forth_keeps_every_pane_and_its_state(app, tmp_path):
    w = MainWindow()
    w.open_paths([str(make_rec(tmp_path / "r"))])
    shown(w, (1914, 1047))
    w.player.show_frame(40)
    w.dash.setCurrentIndex(3); w.dash.append_log("kept line")
    for size in ((1100, 800), (1914, 1047), (1000, 700), (1700, 950)):
        w.resize(*size); pump(0.3)
        assert w.view.isVisible() and w.player.isVisible() and w.dash.isVisible() and w.side_scroll.isVisible()
        assert w.player.idx == 40 and w.dash.currentIndex() == 3 and "kept line" in w.dash.log.toPlainText()
        assert w.view._img is not None
    assert w._wide is True


def test_the_buttons_still_work_in_the_wide_layout(app, tmp_path):
    w = MainWindow(DiscSegmenter(), fake_framework())
    w.open_paths([str(make_rec(tmp_path / "r"))])
    shown(w, (1914, 1047))
    assert w.btn_analyze.isVisible() and w.btn_analyze.isEnabled()
    w.player.btn_play.click(); pump(0.3)
    assert w.player.idx >= 0
    w.player.pause_play()
    w.pick_trap(); w.on_click(80, 60)
    assert w.trap is not None


def test_the_window_can_shrink_to_a_small_screen(app):
    w = MainWindow()
    assert w.minimumSizeHint().width() <= 1250 and w.minimumSizeHint().height() <= 760


def test_f11_toggles_full_screen_and_escape_leaves_it(app):
    w = MainWindow()
    w.show(); pump(0.2)
    w.toggle_fullscreen(); pump(0.3)
    assert w.isFullScreen()
    w.toggle_fullscreen(); pump(0.3)
    assert not w.isFullScreen()

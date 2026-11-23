"""Browsing without analysis: FrameReader, PlayerBar, and the zoom/pan of ImageView. Offscreen."""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("celldform.demo.qt")
from celldform.demo.qt import QtCore, QtGui, QtWidgets

from celldform.demo.gui import ImageView, MainWindow, to_qimage
from celldform.demo.player import PlayerBar
from celldform.demo.source import FrameReader, Recording
from tests.test_demo_run import DiscSegmenter
from tests.test_demo_session import fake_framework

W, H, N = 160, 120, 90


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait(cond, secs=10.0):
    t = time.time()
    while not cond() and time.time() - t < secs:
        QtWidgets.QApplication.processEvents(); time.sleep(0.01)
    QtWidgets.QApplication.processEvents()
    return cond()


def make_rec(folder, n=N, stills=True, log=True):
    """Frame k is a flat image of gray 20 + 2k, so any frame can be identified from its mean."""
    folder.mkdir()
    w = cv2.VideoWriter(str(folder / "video.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 22.0, (W, H), True)
    for k in range(n):
        w.write(np.full((H, W, 3), min(20 + 2 * k, 255), np.uint8))
    w.release()
    if stills:
        d = folder / "stills_1GB"; d.mkdir()
        keep = [k for k in range(n) if k % 5 != 3]                              # some frames have no still
        for k in keep:
            cv2.imwrite(str(d / f"s{k:05d}.png"), np.full((H, W, 3), min(20 + 2 * k, 255), np.uint8))
        pd.DataFrame({"still_file": [f"s{k:05d}.png" for k in keep], "video_frame": [k + 1 for k in keep]}).to_csv(
            folder / "stills.csv", index=False)
    if log:
        pd.DataFrame({"time_s": np.arange(n) / 22.0, "set_current_mA": 350 + 5 * np.arange(n)}).to_csv(folder / "frames.csv", index=False)
    return folder


def level(frame):
    return int(round(float(frame.mean())))


def near(frame, k):
    return abs(level(frame) - (20 + 2 * k)) <= 3


# ── FrameReader ─────────────────────────────────────────────────────────────────────────────────────────────────

def test_video_reader_random_access_in_any_order(tmp_path):
    r = FrameReader(Recording(make_rec(tmp_path / "r"), size=(W, H)))
    assert r.count == N
    for k in (0, 1, 2, 3, 30, 31, 5, 80, 79, 0, 45):                 # forward, short skip, long seek, back, repeat
        assert near(r.read(k), k), k
    assert near(r.read(10_000), N - 1) and near(r.read(-5), 0)       # clamped into range
    assert r.read(0).shape == (H, W, 3)
    r.close()


def test_image_reader_uses_the_nearest_earlier_still_when_one_is_missing(tmp_path):
    rec = Recording(make_rec(tmp_path / "r"), size=(W, H), source="images")
    r = FrameReader(rec)
    assert r.count == N and near(r.read(10), 10)
    assert near(r.read(13), 12)                                        # frame 13 has no still: frame 12's is shown
    assert near(r.read(0), 0)


def test_reader_can_strip_the_overlay_text(tmp_path):
    folder = make_rec(tmp_path / "r", stills=False)
    # a frame with yellow text
    w = cv2.VideoWriter(str(folder / "video.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 22.0, (320, 240), True)
    f = np.full((240, 320, 3), 120, np.uint8)
    cv2.putText(f, "Current: 322 mA", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    for _ in range(3):
        w.write(f)
    w.release()
    rec = Recording(folder, size=(320, 240))
    raw = FrameReader(rec).read(1)
    clean = FrameReader(rec, remove_text=True).read(1)
    yellow = lambda a: int(((a[..., 2] > 180) & (a[..., 1] > 180) & (a[..., 0] < 100)).sum())
    assert yellow(raw) > 200 and yellow(clean) < yellow(raw) * 0.1


# ── PlayerBar ───────────────────────────────────────────────────────────────────────────────────────────────────

def make_player(tmp_path, **kw):
    pl = PlayerBar()
    shown = []
    pl.frame_shown.connect(lambda f, i: shown.append((i, level(f))))
    pl.set_recording(Recording(make_rec(tmp_path / "r", **kw), size=(W, H)))
    return pl, shown


def test_player_starts_on_frame_zero_and_describes_it(app, tmp_path):
    pl, shown = make_player(tmp_path)
    assert pl.isEnabled() and shown[0][0] == 0 and pl.slider.maximum() == N - 1
    assert "frame 0 / 89" in pl.info.text() and "0.0 s" in pl.info.text() and "350 mA" in pl.info.text()
    pl.show_frame(40)
    assert "frame 40" in pl.info.text() and "550 mA" in pl.info.text() and "1.8 s" in pl.info.text()
    assert pl.slider.value() == 40


def test_step_buttons_use_the_chosen_step_size_and_clamp(app, tmp_path):
    pl, shown = make_player(tmp_path)
    pl.step(1)
    assert pl.idx == 1
    pl.step_box.setCurrentIndex([pl.step_box.itemData(i) for i in range(pl.step_box.count())].index(10))
    pl.step(1); pl.step(1)
    assert pl.idx == 21
    pl.step(-1)
    assert pl.idx == 11
    pl.step_box.setCurrentIndex([pl.step_box.itemData(i) for i in range(pl.step_box.count())].index(1000))
    pl.step(1)
    assert pl.idx == N - 1                                           # clamped to the last frame
    pl.step(-1)
    assert pl.idx == 0
    pl.btn_last.click(); assert pl.idx == N - 1
    pl.btn_first.click(); assert pl.idx == 0


def test_slider_scrubs_and_the_frame_follows(app, tmp_path):
    pl, shown = make_player(tmp_path)
    pl.slider.setValue(60)
    assert pl.idx == 60 and near_level(shown[-1][1], 60)


def near_level(lv, k):
    return abs(lv - (20 + 2 * k)) <= 3


def test_playing_advances_with_the_clock_and_stops_at_the_end(app, tmp_path):
    pl, shown = make_player(tmp_path)
    pl.speed.setCurrentIndex([pl.speed.itemData(i) for i in range(pl.speed.count())].index(16.0))
    pl.btn_play.click()
    assert pl.playing and pl.btn_play.text() == "Pause"
    assert wait(lambda: pl.idx > 20)                                  # 16x of 22 fps: ~350 frames a second
    assert wait(lambda: not pl.playing, 5.0)                          # reached the end by itself
    assert pl.idx == N - 1 and pl.btn_play.text() == "Play"
    idxs = [i for i, _ in shown]
    assert idxs == sorted(idxs)                                       # never goes backwards


def test_pressing_play_at_the_end_starts_over_and_pause_holds_the_frame(app, tmp_path):
    pl, shown = make_player(tmp_path)
    pl.btn_last.click()
    pl.speed.setCurrentIndex([pl.speed.itemData(i) for i in range(pl.speed.count())].index(0.25))
    pl.btn_play.click()
    assert wait(lambda: pl.idx >= 1)
    pl.btn_play.click()
    assert not pl.playing
    held = pl.idx
    wait(lambda: False, 0.3)
    assert pl.idx == held


def test_playback_speed_follows_the_speed_box(app, tmp_path):
    def played(speed, secs=0.6):
        pl, _ = make_player(tmp_path / f"s{speed}", n=400)
        pl.speed.setCurrentIndex([pl.speed.itemData(i) for i in range(pl.speed.count())].index(speed))
        pl.btn_play.click()
        wait(lambda: False, secs)
        pl.pause_play()
        return pl.idx
    (tmp_path / "s0.5").mkdir(); (tmp_path / "s4.0").mkdir()
    slow, fast = played(0.5), played(4.0)
    assert fast > 3 * slow and 3 <= slow <= 14                      # 22 fps * 0.5 * 0.6 s ~ 7 frames


def test_browsing_is_off_while_analysing(app, tmp_path):
    pl, _ = make_player(tmp_path)
    pl.play()
    pl.set_analysis_running(True)
    assert not pl.playing and not pl.isEnabled()
    pl.toggle_play(); pl.step(1)
    assert pl.idx == 0 and not pl.playing                             # the controls do nothing while disabled
    pl.set_analysis_running(False)
    assert pl.isEnabled()


def test_an_undecodable_frame_is_reported_and_playback_stops(app, tmp_path):
    pl, shown = make_player(tmp_path)
    pl.reader.read = lambda i: None
    pl.play()
    assert not pl.show_frame(5)
    assert "cannot be decoded" in pl.info.text() and not pl.playing


def test_hide_overlay_checkbox_redraws_the_current_frame(app, tmp_path):
    pl, shown = make_player(tmp_path)
    n = len(shown)
    pl.no_text.setChecked(True)
    assert pl.reader.remove_text and len(shown) == n + 1 and shown[-1][0] == pl.idx


def test_images_source_in_the_player(app, tmp_path):
    folder = make_rec(tmp_path / "r")
    pl = PlayerBar()
    shown = []
    pl.frame_shown.connect(lambda f, i: shown.append((i, level(f))))
    pl.set_recording(Recording(folder, size=(W, H), source="images"))
    pl.show_frame(13)
    assert near_level(shown[-1][1], 12) and pl.idx == 13


# ── ImageView: zoom and pan ─────────────────────────────────────────────────────────────────────────────────────

def make_view(iw=200, ih=100, vw=400, vh=200):
    v = ImageView()
    v.setMinimumSize(0, 0)
    v.resize(vw, vh)
    img = QtGui.QImage(iw, ih, QtGui.QImage.Format.Format_RGB888)
    img.fill(QtGui.QColor(90, 90, 90))
    v.show_image(img)
    return v


def mouse(kind, x, y, button=QtCore.Qt.MouseButton.LeftButton, held=True):
    t = {"press": QtCore.QEvent.Type.MouseButtonPress, "move": QtCore.QEvent.Type.MouseMove,
         "release": QtCore.QEvent.Type.MouseButtonRelease}[kind]
    b = button if kind != "move" else QtCore.Qt.MouseButton.NoButton
    held_b = button if held and kind != "release" else QtCore.Qt.MouseButton.NoButton
    p = QtCore.QPointF(x, y)
    return QtGui.QMouseEvent(t, p, p, b, held_b, QtCore.Qt.KeyboardModifier.NoModifier)


def test_a_click_maps_to_image_pixels_and_ignores_the_bars(app):
    v = make_view(200, 100, 400, 300)                 # scale 2, 50 px bars above and below
    got = []
    v.clicked.connect(lambda x, y: got.append((x, y)))
    for ev in (mouse("press", 100, 150), mouse("release", 100, 150)):
        v.mousePressEvent(ev) if ev.type() == QtCore.QEvent.Type.MouseButtonPress else v.mouseReleaseEvent(ev)
    assert got == [pytest.approx((50, 50))]
    v.mousePressEvent(mouse("press", 100, 10)); v.mouseReleaseEvent(mouse("release", 100, 10))
    assert len(got) == 1                               # in the black bar: no click


def test_zoom_keeps_the_point_under_the_cursor_fixed(app):
    v = make_view()
    before = v.widget_to_image(300, 50)
    v._set_zoom(4.0, (300, 50))
    after = v.widget_to_image(300, 50)
    assert v.zoom == 4.0 and after == pytest.approx(before, abs=0.01)
    assert v.widget_to_image(0, 0)[0] > 100            # now looking at the right-hand part of the image
    v._set_zoom(1000, (10, 10))
    assert v.zoom == v.MAX_ZOOM
    v._set_zoom(0.1, (10, 10))
    assert v.zoom == 1.0 and v.center == (0.5, 0.5)    # zooming out past 1 shows the whole image again


def test_the_wheel_zooms_at_the_cursor(app):
    v = make_view()
    pos = QtCore.QPointF(300, 50)
    ev = QtGui.QWheelEvent(pos, pos, QtCore.QPoint(0, 0), QtCore.QPoint(0, 120), QtCore.Qt.MouseButton.NoButton,
                           QtCore.Qt.KeyboardModifier.NoModifier, QtCore.Qt.ScrollPhase.NoScrollPhase, False)
    v.wheelEvent(ev)
    assert v.zoom == pytest.approx(1.25)


def test_dragging_pans_when_zoomed_and_is_not_a_click(app):
    v = make_view()
    got = []
    v.clicked.connect(lambda x, y: got.append((x, y)))
    v._set_zoom(4.0, (200, 100))                       # centred zoom
    c0 = v.widget_to_image(200, 100)
    v.mousePressEvent(mouse("press", 200, 100))
    v.mouseMoveEvent(mouse("move", 160, 100))          # drag 40 widget px left = 40/(2*4)=5 image px to the right
    v.mouseReleaseEvent(mouse("release", 160, 100))
    c1 = v.widget_to_image(200, 100)
    assert c1[0] == pytest.approx(c0[0] + 5, abs=0.1) and c1[1] == pytest.approx(c0[1], abs=0.1)
    assert got == []                                   # a drag is not a click
    for _ in range(20):                                # drag far: the view stops at the image edge
        v.mousePressEvent(mouse("press", 200, 100)); v.mouseMoveEvent(mouse("move", 0, 100)); v.mouseReleaseEvent(mouse("release", 0, 100))
    s, x0, y0, w, h = v._geometry()
    assert x0 + w == pytest.approx(200, abs=0.01) and x0 >= 0


def test_a_click_while_zoomed_gives_the_zoomed_pixel(app):
    v = make_view()
    got = []
    v.clicked.connect(lambda x, y: got.append((x, y)))
    v._set_zoom(4.0, (200, 100))
    expect = v.widget_to_image(250, 80)
    v.mousePressEvent(mouse("press", 250, 80)); v.mouseReleaseEvent(mouse("release", 250, 80))
    assert got == [pytest.approx(expect)] and 100 < got[0][0] < 130 and 30 < got[0][1] < 60


def test_double_click_and_reset_show_the_whole_image(app):
    v = make_view()
    v._set_zoom(8.0, (300, 150))
    v.mouseDoubleClickEvent(mouse("press", 1, 1))
    assert v.zoom == 1.0 and v.center == (0.5, 0.5)


def test_the_zoom_survives_a_new_picture_even_of_another_size(app):
    v = make_view(200, 100)
    v._set_zoom(3.0, (100, 50))
    z, c = v.zoom, v.center
    img = QtGui.QImage(400, 200, QtGui.QImage.Format.Format_RGB888); img.fill(QtGui.QColor(10, 10, 10))
    v.show_image(img)
    assert v.zoom == z and v.center == c and v.image_size() == (400, 200)


def test_hover_describes_the_pixel_and_painting_works(app):
    v = make_view()
    seen = []
    v.hovered.connect(seen.append)
    v.mouseMoveEvent(mouse("move", 200, 100, held=False))
    assert seen[-1] == "x 100  y 50  gray 90"
    v.mouseMoveEvent(mouse("move", 1, 1000, held=False))
    assert seen[-1] == ""
    v.set_banner("PAUSED"); v._set_zoom(2.0, (10, 10))
    pix = v.grab()                                     # paints the image, the zoom note and the banner
    assert not pix.isNull()
    assert ImageView().grab() is not None             # and an empty view


# ── in the window ───────────────────────────────────────────────────────────────────────────────────────────────

def test_window_browses_without_analysis_and_the_trap_click_uses_analysis_pixels(app, tmp_path):
    w = MainWindow(DiscSegmenter(), fake_framework())
    w.load(str(make_rec(tmp_path / "r")))
    assert w.player.isEnabled() and w.view.image_size() == (W, H)         # frame 0 of the recording is on show
    assert abs(w.view._img.pixelColor(5, 5).red() - 20) <= 3
    w.player.show_frame(40)                                               # browsing changes what the view shows
    assert abs(w.view._img.pixelColor(5, 5).red() - 100) <= 3
    assert w.btn_analyze.isEnabled() and w.verdict_lbl.text() == "Ready"    # and runs no analysis
    w.pick_trap(); w.on_click(80, 60)                                      # the middle of a 160x120 frame ...
    assert w.trap == pytest.approx((614, 461), abs=1)                     # ... is the middle of the 1228x922 analysis frame
    assert "Trap set at (614, 461)" in w.info.text()


def test_window_player_follows_the_frames_source(app, tmp_path):
    w = MainWindow(DiscSegmenter(), fake_framework())
    w.load(str(make_rec(tmp_path / "r", n=450)))
    assert w.player.count == 450 and "frame 0 / 449" in w.player.info.text()
    w.source_box.setCurrentIndex(1)                                        # the stills
    assert w.player.count == 450 and w.player.reader.rec.source == "images"
    w.source_box.setCurrentIndex(0)
    assert w.player.reader.rec.source == "video"


def test_hover_and_state_labels_exist(app):
    w = MainWindow()
    assert w.state_lbl.text() == "Idle"
    w.view.hovered.emit("x 1  y 2  gray 3")
    assert w.pos_lbl.text() == "x 1  y 2  gray 3"


# ── opening a video file, hand-picked images, or a folder ───────────────────────────────────────────────────────

def two_videos(tmp_path, with_stills=False):
    d = tmp_path / "lab"; d.mkdir()
    for name, n in (("a.avi", 30), ("b.avi", 60)):
        w = cv2.VideoWriter(str(d / name), cv2.VideoWriter_fourcc(*"MJPG"), 22.0, (W, H), True)
        for k in range(n):
            w.write(np.full((H, W, 3), 50, np.uint8))
        w.release()
    return d


def test_the_open_menu_offers_video_images_and_folder(app):
    w = MainWindow()
    assert [a.text() for a in w.open_btn.menu().actions()] == ["Video file…", "Image files… (select several)", "Recording folder…"]


def test_a_chosen_video_file_is_the_one_opened(app, tmp_path):
    d = two_videos(tmp_path)
    w = MainWindow(DiscSegmenter(), fake_framework())
    w.open_paths([str(d / "b.avi")])
    assert w.player.reader.rec.video.name == "b.avi" and w.player.count == 60
    assert "b.avi" in w.info.text() and w.btn_analyze.isEnabled() and w.source_box.isEnabled()
    w.open_paths([str(d / "a.avi")])
    assert w.player.count == 30


def test_a_folder_with_several_videos_asks_which(app, tmp_path, monkeypatch):
    d = two_videos(tmp_path)
    asked = []

    def pick(parent, title, label, items, current, editable):
        asked.append(list(items))
        return "b.avi", True
    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", staticmethod(pick))
    w = MainWindow()
    w.open_paths([str(d)])
    assert asked == [["a.avi", "b.avi"]] and w.player.count == 60
    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", staticmethod(lambda *a, **k: ("", False)))
    w2 = MainWindow()
    w2.open_paths([str(d)])                                              # cancelled: nothing opened
    assert w2.recording is None and not w2.btn_analyze.isEnabled()


def test_a_folder_with_one_video_opens_without_asking(app, tmp_path, monkeypatch):
    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", staticmethod(lambda *a, **k: pytest.fail("should not ask")))
    w = MainWindow(DiscSegmenter(), fake_framework())
    w.open_paths([str(make_rec(tmp_path / "r"))])
    assert w.player.count == N and w.btn_analyze.isEnabled()


def test_several_image_files_open_as_one_recording_and_lock_the_source(app, tmp_path):
    folder = make_rec(tmp_path / "r")
    pics = sorted((folder / "stills_1GB").glob("*.png"))[:10]
    w = MainWindow(DiscSegmenter(), fake_framework())
    w.open_paths([str(p) for p in pics])
    assert w.source_box.currentData() == "images" and not w.source_box.isEnabled()
    assert w.player.reader.rec.source == "images" and len(w.player.reader.rec.images) == 10
    assert "10 images from stills_1GB" in w.info.text() and w.btn_analyze.isEnabled()
    w.open_paths([str(folder)])                                           # a folder again: the box is free
    assert w.source_box.isEnabled()


def test_hand_picked_images_can_be_analysed_end_to_end(app, tmp_path, monkeypatch):
    from tests.test_demo_gui import _patched_run
    _patched_run(monkeypatch)
    folder = make_rec(tmp_path / "r", n=450)
    pics = [str(p) for p in sorted((folder / "stills_1GB").glob("*.png"))]
    w = MainWindow(DiscSegmenter(), fake_framework())
    w.open_paths(pics)
    w.btn_analyze.click()
    assert wait(lambda: w.worker.isFinished(), 30)
    log = w.dash.log.toPlainText()
    assert "start: 360 images from stills_1GB" in log or "images from stills_1GB" in log
    assert "source=images" in log and log.count("seg ") > 5 and w.source_box.isEnabled() is False


def test_mixed_or_unusable_selections_are_refused_with_a_reason(app, tmp_path):
    d = two_videos(tmp_path)
    folder = make_rec(tmp_path / "r")
    pic = str(next((folder / "stills_1GB").glob("*.png")))
    (tmp_path / "notes.txt").write_text("x")
    w = MainWindow()
    w.open_paths([str(d / "a.avi"), pic])
    assert w.verdict_lbl.text().startswith("Cannot open") and "not a mix" in w.verdict_lbl.text()
    w.open_paths([str(tmp_path / "notes.txt")])
    assert "neither a video nor an image" in w.verdict_lbl.text()
    w.open_paths([str(d / "a.avi"), str(d / "b.avi")])
    assert w.verdict_lbl.text().startswith("Cannot open") and w.recording is None
    w.open_paths([])                                                      # nothing chosen: nothing happens
    assert w.recording is None


def test_a_folder_of_pictures_with_no_video_is_read_as_images(app, tmp_path):
    d = tmp_path / "pics"; d.mkdir()
    for k in range(4):
        cv2.imwrite(str(d / f"p{k}.png"), np.full((H, W, 3), 30 * k, np.uint8))
    w = MainWindow(DiscSegmenter(), fake_framework())
    w.open_paths([str(d)])
    assert w.source_box.currentData() == "images" and w.player.count == 4 and w.btn_analyze.isEnabled()


def test_dropping_several_files_opens_them(app, tmp_path):
    folder = make_rec(tmp_path / "r")
    pics = sorted((folder / "stills_1GB").glob("*.png"))[:6]
    mime = QtCore.QMimeData()
    mime.setUrls([QtCore.QUrl.fromLocalFile(str(p)) for p in pics])
    ev = QtGui.QDropEvent(QtCore.QPointF(5, 5), QtCore.Qt.DropAction.CopyAction, mime, QtCore.Qt.MouseButton.LeftButton,
                          QtCore.Qt.KeyboardModifier.NoModifier)
    w = MainWindow()
    w.dropEvent(ev)
    assert len(w.player.reader.rec.images) == 6
    mime2 = QtCore.QMimeData()
    mime2.setUrls([QtCore.QUrl.fromLocalFile(str(folder / "video.avi"))])
    w.dropEvent(QtGui.QDropEvent(QtCore.QPointF(5, 5), QtCore.Qt.DropAction.CopyAction, mime2,
                                 QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier))
    assert w.player.reader.rec.source == "video" and w.player.count == N


def test_menu_actions_use_the_file_dialogs(app, tmp_path, monkeypatch):
    d = two_videos(tmp_path)
    folder = make_rec(tmp_path / "r")
    pics = [str(p) for p in sorted((folder / "stills_1GB").glob("*.png"))[:3]]
    w = MainWindow()
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(d / "b.avi"), "")))
    w.act_open_video.trigger()
    assert w.player.count == 60
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames", staticmethod(lambda *a, **k: (pics, "")))
    w.act_open_images.trigger()
    assert len(w.player.reader.rec.images) == 3
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(folder)))
    w.act_open_folder.trigger()
    assert w.player.count == N
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: ("", "")))
    before = w.player.count
    w.act_open_video.trigger()                                            # cancelled dialog: nothing changes
    assert w.player.count == before


def test_a_video_files_stills_can_be_used_by_switching_the_source(app, tmp_path):
    folder = make_rec(tmp_path / "r")
    w = MainWindow()
    w.open_paths([str(folder / "video.avi")])
    w.source_box.setCurrentIndex(1)                                       # the stills next to this video
    assert w.player.reader.rec.source == "images" and len(w.player.reader.rec.images) > 0
    w.source_box.setCurrentIndex(0)
    assert w.player.reader.rec.source == "video" and w.player.reader.rec.video.name == "video.avi"

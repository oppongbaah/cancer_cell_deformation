"""
Desktop demo (PySide6): open a recording, optionally click the trapped cell, run, watch the masks and the HER2 result
build up. It drives the same :func:`celldform.demo.run.run` as the command line, in a worker thread.

    python -m celldform.demo.gui [recording]

Not real-time: frames are measured as fast as the models allow (a measured frame costs a network pass; the rest of the
video is only skipped through). The HER2 line stays "insufficient" until 10 laser steps (~91 s of video) are measured.
"""
from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from celldform.demo.qt import QtCore, QtGui, QtWidgets

from celldform.demo import dashboard, privacy
from celldform.demo import models as M
from celldform.demo.settings import Settings
from celldform.demo.player import PlayerBar
from celldform.demo.run import run
from celldform.demo.source import FEATURE_SIZE


def map_click(pos: Tuple[float, float], label_size: Tuple[int, int], image_size: Tuple[int, int] = FEATURE_SIZE
              ) -> Optional[Tuple[float, float]]:
    """Widget position of a click on an aspect-preserving, centred image -> image pixel; None if outside the image."""
    lw, lh = label_size
    iw, ih = image_size
    s = min(lw / iw, lh / ih)
    dw, dh = iw * s, ih * s
    x, y = pos[0] - (lw - dw) / 2, pos[1] - (lh - dh) / 2
    if not (0 <= x < dw and 0 <= y < dh):
        return None
    return x / s, y / s


def to_qimage(bgr: np.ndarray) -> QtGui.QImage:
    h, w = bgr.shape[:2]
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    return QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()


class ImageView(QtWidgets.QWidget):
    """The picture, with zoom and pan.

    Mouse wheel zooms at the cursor, dragging pans, double-click shows the whole image again. A click that did not
    drag emits ``clicked(x, y)`` in image pixels (used to mark the trapped cell). ``hovered(text)`` describes the
    pixel under the cursor. The view (zoom, centre) is kept while the picture changes, so playing or analysing a
    recording does not undo the zoom.
    """
    clicked = QtCore.Signal(float, float)
    hovered = QtCore.Signal(str)
    MAX_ZOOM = 32.0

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self._img: Optional[QtGui.QImage] = None
        self.zoom, self.center = 1.0, (0.5, 0.5)           # centre as a fraction of the image, so it survives a size change
        self.banner = ""
        self._press: Optional[Tuple[float, float, object]] = None
        self._dragged = False

    # -- picture and view ---------------------------------------------------------------------------------------
    def show_image(self, img: QtGui.QImage) -> None:
        self._img = img
        self.update()

    def image_size(self) -> Tuple[int, int]:
        return (self._img.width(), self._img.height()) if self._img else FEATURE_SIZE

    def reset_view(self) -> None:
        self.zoom, self.center = 1.0, (0.5, 0.5)
        self.update()

    def set_banner(self, text: str) -> None:
        self.banner = text
        self.update()

    def _geometry(self):
        """(scale, x0, y0, w, h): widget pixels per image pixel, and the visible window in image pixels."""
        iw, ih = self.image_size()
        lw, lh = max(self.width(), 1), max(self.height(), 1)
        s = min(lw / iw, lh / ih) * self.zoom
        w, h = lw / s, lh / s
        x0, y0 = self.center[0] * iw - w / 2, self.center[1] * ih - h / 2
        x0 = -(w - iw) / 2 if w >= iw else min(max(x0, 0.0), iw - w)
        y0 = -(h - ih) / 2 if h >= ih else min(max(y0, 0.0), ih - h)
        return s, x0, y0, w, h

    def widget_to_image(self, x: float, y: float, inside_only: bool = True) -> Optional[Tuple[float, float]]:
        s, x0, y0, _, _ = self._geometry()
        iw, ih = self.image_size()
        ix, iy = x0 + x / s, y0 + y / s
        if inside_only and not (0 <= ix < iw and 0 <= iy < ih):
            return None
        return ix, iy

    # -- drawing ------------------------------------------------------------------------------------------------
    def paintEvent(self, e) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#111"))
        if self._img is None:
            p.setPen(QtGui.QColor("#888"))
            p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "Open a recording")
            return
        s, x0, y0, w, h = self._geometry()
        iw, ih = self.image_size()
        sx0, sy0, sx1, sy1 = max(x0, 0.0), max(y0, 0.0), min(x0 + w, iw), min(y0 + h, ih)
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, s < 3)       # zoomed in: show the real pixels
        p.drawImage(QtCore.QRectF((sx0 - x0) * s, (sy0 - y0) * s, (sx1 - sx0) * s, (sy1 - sy0) * s), self._img,
                    QtCore.QRectF(sx0, sy0, sx1 - sx0, sy1 - sy0))
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        font = p.font(); font.setBold(True); font.setPointSize(11); p.setFont(font)
        if self.zoom > 1.001:
            p.setPen(QtGui.QColor("#ddd"))
            p.drawText(10, self.height() - 10, f"{self.zoom:.1f}x   double-click to fit")
        if self.banner:
            font.setPointSize(16); p.setFont(font)
            fm = QtGui.QFontMetrics(font)
            tw, th = fm.horizontalAdvance(self.banner) + 28, fm.height() + 10
            r = QtCore.QRectF((self.width() - tw) / 2, 12, tw, th)
            p.setPen(QtCore.Qt.PenStyle.NoPen); p.setBrush(QtGui.QColor(224, 160, 48, 235)); p.drawRoundedRect(r, 6, 6)
            p.setPen(QtGui.QColor("black")); p.drawText(r, QtCore.Qt.AlignmentFlag.AlignCenter, self.banner)

    # -- mouse --------------------------------------------------------------------------------------------------
    def _set_zoom(self, zoom: float, anchor: Tuple[float, float]) -> None:
        """Zoom keeping the image point under ``anchor`` (widget pixels) where it is."""
        pt = self.widget_to_image(anchor[0], anchor[1], inside_only=False)
        iw, ih = self.image_size()
        zoom = min(max(zoom, 1.0), self.MAX_ZOOM)
        if zoom == 1.0 or pt is None:
            self.reset_view()
            return
        self.zoom = zoom
        s = min(max(self.width(), 1) / iw, max(self.height(), 1) / ih) * zoom
        cx = pt[0] - (anchor[0] - self.width() / 2) / s
        cy = pt[1] - (anchor[1] - self.height() / 2) / s
        self.center = (cx / iw, cy / ih)
        self.update()

    def wheelEvent(self, e) -> None:  # noqa: N802
        steps = e.angleDelta().y() / 120.0
        if steps:
            self._set_zoom(self.zoom * (1.25 ** steps), (e.position().x(), e.position().y()))

    def mousePressEvent(self, e) -> None:  # noqa: N802
        self._press, self._dragged = (e.position().x(), e.position().y(), e.button()), False

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        x, y = e.position().x(), e.position().y()
        if self._press is not None and e.buttons():
            px, py, _ = self._press
            if self._dragged or abs(x - px) + abs(y - py) > 4:
                self._dragged = True
                if self.zoom > 1.0 and self._img is not None:
                    s, x0, y0, w, h = self._geometry()
                    iw, ih = self.image_size()
                    self.center = ((x0 + w / 2 - (x - px) / s) / iw, (y0 + h / 2 - (y - py) / s) / ih)
                    self._press = (x, y, self._press[2])
                    self.update()
            return
        p = self.widget_to_image(x, y)
        if p is not None and self._img is not None:
            ix, iy = int(p[0]), int(p[1])
            self.hovered.emit(f"x {ix}  y {iy}  gray {QtGui.qGray(self._img.pixel(ix, iy))}")
        else:
            self.hovered.emit("")

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if self._press is not None and not self._dragged and self._press[2] == QtCore.Qt.MouseButton.LeftButton:
            p = self.widget_to_image(e.position().x(), e.position().y())
            if p:
                self.clicked.emit(*p)
        self._press = None

    def mouseDoubleClickEvent(self, e) -> None:  # noqa: N802
        self.reset_view()


class DashboardTabs(QtWidgets.QTabWidget):
    """The dashboard pages (celldform.demo.dashboard), one matplotlib canvas per tab; only the visible one is redrawn."""

    def __init__(self) -> None:
        super().__init__()
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self.state = dashboard.State()
        self.canvases = {}
        for page in dashboard.PAGES:
            canvas = FigureCanvasQTAgg(Figure(figsize=(9, 3.2), dpi=100, facecolor="white"))
            self.canvases[page] = canvas
            self.addTab(canvas, page)
        self.log = QtWidgets.QPlainTextEdit()                 # the inference log: one line per event / measured frame
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(20000)
        self.log.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.log.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.log.setPlaceholderText("The inference log appears here while a recording runs.")
        self.addTab(self.log, "Log")
        self.currentChanged.connect(lambda _: self.redraw())

    def append_log(self, line: str) -> None:
        self.log.appendPlainText(line)

    def set_state(self, state) -> None:
        self.state = state
        self.redraw()

    def redraw(self) -> None:
        page = self.tabText(self.currentIndex())
        if page not in self.canvases:                         # the log tab is text, nothing to draw
            return
        canvas = self.canvases[page]
        dashboard.draw_page(canvas.figure, self.state, page)
        canvas.draw_idle()


class Latest:
    """Hand the newest item from a worker thread to the window without queueing.

    The window takes whatever is newest when it gets round to it, so a slow window shows the current frame instead of
    replaying a backlog (which made Pause look as if it did nothing: the video kept playing for seconds).
    """

    def __init__(self) -> None:
        self._lock, self._item, self._pending = threading.Lock(), None, False

    def put(self, item) -> bool:
        """Store ``item``; True when the window has to be told (it is not already on its way to collect one)."""
        with self._lock:
            self._item = item
            need, self._pending = not self._pending, True
        return need

    def take(self):
        with self._lock:
            item, self._item, self._pending = self._item, None, False
        return item


class LogBuffer:
    """Log lines from the worker, delivered to the window in batches and never dropped."""

    def __init__(self) -> None:
        self._lock, self._lines, self._pending = threading.Lock(), [], False

    def put(self, line: str) -> bool:
        with self._lock:
            self._lines.append(line)
            need, self._pending = not self._pending, True
        return need

    def take(self) -> list:
        with self._lock:
            lines, self._lines, self._pending = self._lines, [], False
        return lines


class Worker(QtCore.QThread):
    """Runs the models over a recording. Frames, dashboard snapshots and log lines reach the window through
    :class:`Latest` / :class:`LogBuffer` (the ``*_ready`` signals only say "something is waiting")."""
    frame_ready = QtCore.Signal()
    snapshot_ready = QtCore.Signal()
    log_ready = QtCore.Signal()
    failed = QtCore.Signal(str)
    status = QtCore.Signal(str)

    def __init__(self, recording, segmenter, framework, trap, out, segmenter_name: str = "unet",
                 source: str = "video", model_file: Optional[str] = None, config_file: Optional[str] = None,
                 provisional: bool = False, reference_file: Optional[str] = None) -> None:
        super().__init__()
        self.source, self.provisional, self.reference_file = source, provisional, reference_file
        self.model_file, self.config_file = model_file, config_file
        self._paused, self.paused_s = False, 0.0
        self.recording, self.segmenter, self.framework, self.trap, self.out = recording, segmenter, framework, trap, out
        self.segmenter_name = segmenter_name
        self._stop = False
        self.verdict = None
        self.last_state = None
        self.frames, self.snaps, self.logs = Latest(), Latest(), LogBuffer()

    # -- control (called from the window's thread) ----------------------------------------------------------------
    def cancel(self) -> None:
        self._stop = True
        self._paused = False                                  # a paused run must be able to stop

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    @property
    def paused(self) -> bool:
        return self._paused

    # -- from the worker's thread -----------------------------------------------------------------------------------
    def _say(self, line: str) -> None:
        if self.logs.put(line):
            self.log_ready.emit()

    def _stamp(self, msg: str) -> None:
        self._say(f"{time.strftime('%H:%M:%S')}  {msg}")

    def _wait_if_paused(self) -> None:
        """Called for every frame the source steps over, so the run holds between any two frames."""
        if not self._paused or self._stop:
            return
        t = time.perf_counter()
        self._stamp("paused")
        while self._paused and not self._stop:
            self.msleep(20)
        self.paused_s += time.perf_counter() - t
        if not self._stop:
            self._stamp(f"resumed after {time.perf_counter() - t:.1f} s")

    def run(self) -> None:  # noqa: A003
        try:
            if self.segmenter is None:                       # built here, so the window does not freeze
                if not self.model_file:
                    raise ValueError("no U-Net is loaded: load the U-Net weights and their settings first")
                from celldform.demo.segment import UNetSegmenter
                self.status.emit("Loading the U-Net...")
                t_load = time.perf_counter()
                self._stamp(f"loading the U-Net from {privacy.short_name(self.model_file)}")
                self.segmenter = UNetSegmenter.from_file(self.model_file, self.config_file)
                self._stamp(f"model loaded in {time.perf_counter() - t_load:.1f} s"
                            + (f" ({self.segmenter.describe()})" if hasattr(self.segmenter, "describe") else ""))
            from celldform.demo.source import Recording
            rec = Recording.from_spec(self.recording, source=self.source)
            ref, t0, seen, last_snap = dashboard.load_reference(self.reference_file) if self.reference_file else None, time.perf_counter(), [0], [0.0]
            dev = getattr(self.segmenter, "device", "")

            def snap(sess, n):
                self.last_state = dashboard.build_state(sess, rec, n, time.perf_counter() - t0 - self.paused_s,
                                                        self.segmenter_name, dev, ref)
                if self.snaps.put(self.last_state):
                    self.snapshot_ready.emit()
                last_snap[0] = time.perf_counter()

            def on_frame(i, img, res, v, sess):
                seen[0] += 1
                if self.frames.put((to_qimage(img), v, i)):
                    self.frame_ready.emit()
                if seen[0] == 1 or time.perf_counter() - last_snap[0] > 0.5:       # the dashboard redraw is the slow part
                    snap(sess, seen[0])
                self._sess = sess

            self.verdict = run(self.recording, self.out, self.segmenter, self.framework, write_video=False,
                               segmenter_name=self.segmenter_name, trap_xy=self.trap, stop=lambda: self._stop,
                               on_frame=on_frame, source=self.source, log=self._say, tick=self._wait_if_paused,
                               paused_s=lambda: self.paused_s, provisional=self.provisional)
            if getattr(self, "_sess", None) is not None:
                snap(self._sess, seen[0])                     # the final picture, complete
        except Exception as e:  # shown in the window, not a traceback in a terminal
            self._stamp(f"ERROR  {type(e).__name__}: {e}")
            self.failed.emit(f"{type(e).__name__}: {e}")


class FileRow(QtWidgets.QWidget):
    """One required file: a status mark, its name, what was found in it, and a button to load it."""
    load_clicked = QtCore.Signal(str)

    MARK = {"ok": ("✔", "#2e7d32"), "bad": ("✘", "#c62828"), "empty": ("○", "#888888"), "optional": ("–", "#888888")}

    def __init__(self, role: M.Role) -> None:
        super().__init__()
        self.role = role
        self.mark = QtWidgets.QLabel("○")
        self.mark.setFixedWidth(18)
        self.mark.setStyleSheet("font-size:16px;font-weight:bold;")
        self.title = QtWidgets.QLabel(role.title + ("" if role.required else " (optional)"))
        self.title.setStyleSheet("font-weight:bold;")
        self.detail = QtWidgets.QLabel(role.hint)
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet("color:#666;font-size:11px;")
        self.button = QtWidgets.QPushButton("Load…")
        self.button.setFixedWidth(70)
        self.button.clicked.connect(lambda: self.load_clicked.emit(role.key))
        text = QtWidgets.QVBoxLayout(); text.setSpacing(0)
        text.addWidget(self.title); text.addWidget(self.detail)
        row = QtWidgets.QHBoxLayout(self); row.setContentsMargins(0, 2, 0, 2)
        row.addWidget(self.mark, 0, QtCore.Qt.AlignmentFlag.AlignTop); row.addLayout(text, 1)
        row.addWidget(self.button, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        self.set_check(None)

    def set_check(self, c: Optional[M.Check], note: str = "") -> None:
        kind = "optional" if (c is None and not self.role.required) else "empty" if c is None else "ok" if c.ok else "bad"
        glyph, colour = self.MARK[kind]
        self.mark.setText(glyph)
        self.mark.setStyleSheet(f"font-size:16px;font-weight:bold;color:{colour};")
        if c is None:
            self.detail.setText(self.role.hint)
            self.detail.setStyleSheet("color:#666;font-size:11px;")
            self.button.setText("Load…")
            return
        name = c.path.name if c.path else ""
        extra = f"  ({c.size / 1e6:.1f} MB, SHA-256 {c.short_hash}…)" if c.ok and c.size >= 100_000 else (f"  (SHA-256 {c.short_hash}…)" if c.ok else "")
        self.detail.setText((f"{name}: " if name else "") + c.message + extra + (f"\n{note}" if note else ""))
        self.detail.setStyleSheet(f"color:{'#2e5d32' if c.ok else '#c62828'};font-size:11px;")
        self.button.setText("Change…" if c.ok else "Load…")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, segmenter=None, framework=None, recording: Optional[str] = None,
                 settings: Optional[Settings] = None, results_dir: Optional[Path | str] = None) -> None:
        """``segmenter`` / ``framework`` hand in ready models (tests); a real session loads files through the Required
        files box. ``settings`` remembers the loaded files between sessions (None: nothing is remembered)."""
        super().__init__()
        self.setWindowTitle("celldform - HER2 class of a trapped SKBR3 cell")
        self.settings = settings or Settings()
        self.segmenter, self.framework = segmenter, framework
        self.segmenter_name = "unet" if segmenter is not None else "U-Net"
        self.model_checks: dict = {}                          # role -> models.Check of the file now in use
        self.model_paths: dict = {}
        self.recording, self.trap, self.worker, self.picking = None, None, None, False
        self._results_base = Path(results_dir) if results_dir else Path(tempfile.mkdtemp(prefix="celldform_demo_"))
        self.out = self._results_base                         # each Analyze writes into its own folder below this
        if segmenter is not None:
            self.model_checks["unet"] = M.Check("unet", None, True, "provided by the program")
            self.model_checks["unet_config"] = M.Check("unet_config", None, True, "provided by the program")
        if framework is not None:
            self.model_checks["classifier"] = M.Check("classifier", None, True, "provided by the program")
        self.view = ImageView()
        self.view.clicked.connect(self.on_click)
        self.player = PlayerBar()
        self.player.frame_shown.connect(self.on_browse_frame)
        self.player.fit_requested.connect(self.view.reset_view)
        self.verdict_lbl = QtWidgets.QLabel("Open a recording")
        self.verdict_lbl.setStyleSheet("font-size:20px;font-weight:bold;")
        self.verdict_lbl.setWordWrap(True)
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        self.caveats = QtWidgets.QLabel("")
        self.caveats.setWordWrap(True)
        self.caveats.setStyleSheet("color:#a05a00;")
        self.btn_analyze = QtWidgets.QPushButton("Analyze")
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        for b, tip, css in ((self.btn_analyze, "Analyze the recording: run the models over it (the Browse bar below only looks, it does not analyze)", "background:#2e7d32;color:white;"),
                            (self.btn_pause, "Pause the analysis; press again to resume", "background:#e0a030;color:black;"),
                            (self.btn_stop, "Stop the analysis (the results so far are kept)", "background:#c62828;color:white;")):
            b.setToolTip(tip)
            b.setMinimumHeight(38)
            b.setStyleSheet("QPushButton{font-weight:bold;border-radius:4px;" + css + "}QPushButton:disabled{background:#cfcfcf;color:#888;}")
        self.btn_analyze.clicked.connect(self.start)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_stop.clicked.connect(self.stop)
        buttons = QtWidgets.QHBoxLayout()
        for b in (self.btn_analyze, self.btn_pause, self.btn_stop):
            buttons.addWidget(b)
        self.chk_provisional = QtWidgets.QCheckBox("Give a provisional verdict even if the checks fail")
        self.chk_provisional.setToolTip(
            "Off by default. The classifier was validated only on cells that pass the frame checks and have all 10 laser "
            "steps. With this on, a verdict is also computed from whatever steps passed, but it is always labelled "
            "PROVISIONAL, NOT VALIDATED and is not a HER2 result.")
        self.chk_provisional.setStyleSheet("QCheckBox{color:#a05a00;}")
        self.models_box = QtWidgets.QGroupBox("Required files")
        mv = QtWidgets.QVBoxLayout(self.models_box)
        self.rows = {key: FileRow(role) for key, role in M.ROLES.items()}
        for row in self.rows.values():
            row.load_clicked.connect(self.choose_role)
            mv.addWidget(row)
        self.setup_lbl = QtWidgets.QLabel()
        self.setup_lbl.setWordWrap(True)
        self.btn_bundle = QtWidgets.QToolButton()
        self.btn_bundle.setText("Load a model bundle ▾")
        self.btn_bundle.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_bundle.setToolTip("One folder or .zip that carries all the files, with a checksum for each: any file that "
                                   "was damaged or changed on the way is refused.")
        bundle_menu = QtWidgets.QMenu(self.btn_bundle)
        self.act_bundle_zip = bundle_menu.addAction("Bundle .zip file…", self.choose_bundle_zip)
        self.act_bundle_dir = bundle_menu.addAction("Bundle folder…", self.choose_bundle_folder)
        self.btn_bundle.setMenu(bundle_menu)
        self.btn_clear = QtWidgets.QPushButton("Clear all")
        self.btn_clear.setToolTip("Forget the loaded files (the files themselves are not touched).")
        self.btn_clear.clicked.connect(self.clear_models)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(self.btn_bundle, 1); bar.addWidget(self.btn_clear)
        mv.addLayout(bar)
        mv.addWidget(self.setup_lbl)
        side = QtWidgets.QVBoxLayout()
        side.addLayout(buttons)
        side.addWidget(self.chk_provisional)
        side.addWidget(self.models_box)
        for w in (self.verdict_lbl, self.info, self.caveats):
            side.addWidget(w)
        side.addStretch()
        inner = QtWidgets.QWidget(); inner.setLayout(side)
        sw = QtWidgets.QScrollArea()                          # nothing overlaps when the panel is taller than the room
        sw.setWidget(inner); sw.setWidgetResizable(True); sw.setFixedWidth(360)
        sw.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        sw.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.video_pane = QtWidgets.QWidget()                 # the picture and the Browse bar
        vp = QtWidgets.QVBoxLayout(self.video_pane); vp.setContentsMargins(0, 0, 0, 0)
        vp.addWidget(self.view, 1); vp.addWidget(self.player)
        self.side_scroll = sw
        self.dash = DashboardTabs()
        self.dash.setMinimumHeight(220)
        self._build_layouts()
        self.bar = QtWidgets.QProgressBar(); self.bar.setMaximumWidth(260); self.bar.setValue(0)
        self.state_lbl = QtWidgets.QLabel("Idle")
        self.pos_lbl = QtWidgets.QLabel("")
        self.pos_lbl.setMinimumWidth(190)
        self.view.hovered.connect(self.pos_lbl.setText)
        for w in (self.pos_lbl, self.state_lbl, self.bar):
            self.statusBar().addPermanentWidget(w)
        self.statusBar().showMessage("Open a video file, image files or a recording folder (or drop them on the window).")
        self.setAcceptDrops(True)
        self.total = 0
        self._images_only = False
        self._run_state, self._source, self._retired = "idle", "video", []   # idle | running | paused | stopping
        tb = self.addToolBar("main")
        self.open_btn = QtWidgets.QToolButton()
        self.open_btn.setText("Open recording ▾")
        self.open_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.open_btn.setToolTip("Open one video file, several image files, or a recording folder. "
                                 "You can also drop them on the window.")
        menu = QtWidgets.QMenu(self.open_btn)
        self.act_open_video = menu.addAction("Video file…", self.choose_video)
        self.act_open_images = menu.addAction("Image files… (select several)", self.choose_images)
        self.act_open_folder = menu.addAction("Recording folder…", self.choose)
        self.open_btn.setMenu(menu)
        tb.addWidget(self.open_btn)
        self.act_trap = tb.addAction("Click the cell", self.pick_trap)
        self.act_results = tb.addAction("Open results folder", self.open_results)
        self.act_save = tb.addAction("Save dashboard image…", self.save_dashboard)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(" Frames from: "))
        self.source_box = QtWidgets.QComboBox()
        self.source_box.addItem("Video", "video")
        self.source_box.addItem("Images (stills)", "images")
        self.source_box.setToolTip("Read the video file, or the still images saved next to it (stills*/ folder, or a "
                                   "folder of images). Laser steps and times are the same for both.")
        self.source_box.currentIndexChanged.connect(self.source_changed)
        tb.addWidget(self.source_box)
        self._refresh_models_ui()
        for key, fn in (("Space", self.player.toggle_play), ("Left", lambda: self.player.step(-1)),
                        ("Right", lambda: self.player.step(1)),
                        ("Home", lambda: self.player.isEnabled() and self.player.show_frame(0)),
                        ("End", lambda: self.player.isEnabled() and self.player.show_frame(self.player.count - 1))):
            QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=fn)
        QtGui.QShortcut(QtGui.QKeySequence("F11"), self, activated=self.toggle_fullscreen)
        QtGui.QShortcut(QtGui.QKeySequence("Esc"), self, activated=lambda: self.showNormal() if self.isFullScreen() else None)
        self._refresh_controls()
        if recording:
            self.load(recording)

    # -- one place decides what can be pressed --------------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self.recording is not None and self.player.reader is not None

    def _set_run_state(self, state: str) -> None:
        self._run_state = state
        self._refresh_controls()

    def _refresh_controls(self) -> None:
        """Everything that can be pressed follows from two facts: is a recording loaded, and what the analysis is doing
        (idle, running, paused, stopping). Called after every change, so no stage is left with a stale button."""
        st = self._run_state
        idle, active = st == "idle", st in ("running", "paused")
        self.btn_analyze.setEnabled(self.loaded and idle and self.models_ready)
        missing = M.missing_roles(self.model_checks)
        self.btn_analyze.setToolTip("Open a recording first" if not self.loaded else
                                    ("Load the required files first: " + ", ".join(missing)) if missing else
                                    "Analyze the recording: run the models over it (the Browse bar below only looks, it does not analyze)")
        self.btn_pause.setEnabled(active)
        self.btn_pause.setText("Resume" if st == "paused" else "Pause")
        self.btn_stop.setEnabled(active)
        self.act_trap.setEnabled(self.loaded and idle)
        self.open_btn.setEnabled(idle)
        self.models_box.setEnabled(idle)
        self.chk_provisional.setEnabled(idle)
        self.source_box.setEnabled(idle and not self._images_only)
        self.player.set_analysis_running(not idle)
        self.view.set_banner("PAUSED" if st == "paused" else "")
        self.state_lbl.setText({"idle": "Idle", "running": "Analyzing", "paused": "Paused", "stopping": "Stopping"}[st])

    def report_error(self, where: str, exc: BaseException) -> None:
        """An unexpected error is shown in the status bar and the Log tab; it never takes the window down."""
        msg = f"{where}: {type(exc).__name__}: {exc}"
        self.statusBar().showMessage(msg)
        self.dash.append_log(f"{time.strftime('%H:%M:%S')}  ERROR  {msg}")
        if self.isVisible():
            QtWidgets.QMessageBox.warning(self, "Something went wrong", msg)

    def toggle_fullscreen(self) -> None:
        """Maximize keeps the system's title bar; F11 takes the whole screen (Esc or F11 again to leave)."""
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    # -- layout: fills whatever size the window has --------------------------------------------------------------
    @staticmethod
    def _slot() -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        return w

    def _build_layouts(self) -> None:
        """Two arrangements of the same three panes. Stacked (small or tall windows): picture and controls on top, the
        dashboard below. Wide (maximized, full screen): the picture takes the whole height on the left, the controls and
        the dashboard share the right, so a big screen is used instead of showing black bars."""
        H, V = QtCore.Qt.Orientation.Horizontal, QtCore.Qt.Orientation.Vertical
        self._slots = {k: self._slot() for k in ("video_a", "side_a", "dash_a", "video_b", "side_b", "dash_b")}
        top = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(top); row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._slots["video_a"], 1); row.addWidget(self._slots["side_a"])
        self.page_stacked = QtWidgets.QSplitter(V)
        self.page_stacked.addWidget(top); self.page_stacked.addWidget(self._slots["dash_a"])
        self.page_stacked.setSizes([620, 460])
        right = QtWidgets.QSplitter(V)
        right.addWidget(self._slots["side_b"]); right.addWidget(self._slots["dash_b"])
        right.setSizes([380, 620])
        self.page_wide = QtWidgets.QSplitter(H)
        self.page_wide.addWidget(self._slots["video_b"]); self.page_wide.addWidget(right)
        self.page_wide.setSizes([1180, 740])
        self.page_right = right
        self.pages = QtWidgets.QStackedWidget()
        self.pages.addWidget(self.page_stacked); self.pages.addWidget(self.page_wide)
        self.setCentralWidget(self.pages)
        self._wide = None
        self._arrange(False)

    def _arrange(self, wide: bool) -> None:
        """Move the three panes into the wide or the stacked arrangement."""
        if wide == self._wide:
            return
        self._wide = wide
        k = "b" if wide else "a"
        for pane, name in ((self.video_pane, "video"), (self.side_scroll, "side"), (self.dash, "dash")):
            self._slots[f"{name}_{k}"].layout().addWidget(pane)           # re-parents from the other arrangement
        if wide:
            self.side_scroll.setMinimumWidth(0); self.side_scroll.setMaximumWidth(16777215)
        else:
            self.side_scroll.setFixedWidth(360)
        self.pages.setCurrentWidget(self.page_wide if wide else self.page_stacked)
        QtCore.QTimer.singleShot(0, self._balance)

    def _balance(self) -> None:
        """Give the picture the room its shape needs: the whole height on a wide screen."""
        if not self._wide:
            return
        w, h = self.pages.width(), self.pages.height()
        video_w = int(min(0.64 * w, (h - self.player.sizeHint().height() - 12) * 4 / 3 + 12))
        self.page_wide.setSizes([max(video_w, 480), max(w - video_w, 300)])
        self.page_right.setSizes([int(0.42 * h), int(0.58 * h)])

    @staticmethod
    def wants_wide(width: int, height: int, currently_wide: bool) -> bool:
        """Wide arrangement for landscape windows that are big enough; a margin stops it flipping at the border."""
        ratio = width / max(height, 1)
        return (width >= 1350 and ratio >= 1.3) if currently_wide else (width >= 1500 and ratio >= 1.5)

    def resizeEvent(self, e) -> None:  # noqa: N802
        super().resizeEvent(e)
        if hasattr(self, "pages"):
            self._arrange(self.wants_wide(self.width(), self.height(), bool(self._wide)))

    def dragEnterEvent(self, e) -> None:  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:  # noqa: N802
        self.open_paths([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])

    def save_dashboard(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save dashboard", str(self.out / "dashboard.png"),
                                                        "PNG image (*.png)")
        if path:
            dashboard.save(self.dash.state, path)
            self.statusBar().showMessage(f"Dashboard saved to {path}")

    def open_results(self) -> None:
        folder = self.out if self.out.exists() else self._results_base
        folder.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))

    # -- recording -----------------------------------------------------------------------------------------------
    def choose(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Recording folder", self.settings.last_dir)
        if d:
            self.open_paths([d])

    def choose_video(self) -> None:
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Video file", self.settings.last_dir, "Video files (*.avi *.mp4 *.mov *.mkv *.mpg *.mpeg *.m4v *.wmv);;All files (*)")
        if f:
            self.open_paths([f])

    def choose_images(self) -> None:
        fs, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Image files (select several)", self.settings.last_dir, "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)")
        if fs:
            self.open_paths(fs)

    def open_paths(self, paths) -> None:
        """What the Open menu and drag-and-drop lead to: one video file, one folder, or several image files."""
        from celldform.demo.source import is_image, is_video
        paths = [str(p) for p in paths if p]
        if not paths:
            return
        if self._run_state != "idle":
            self.statusBar().showMessage("Stop the analysis before opening another recording.")
            return
        self.settings.last_dir = str(Path(paths[0]) if Path(paths[0]).is_dir() else Path(paths[0]).parent)
        if len(paths) == 1:
            p = Path(paths[0])
            if p.is_dir():
                return self.open_folder(p)
            if is_video(p):
                return self.load(str(p))
            if is_image(p):
                return self.load([str(p)])
            return self._cannot_open(f"{p.name} is neither a video nor an image")
        if all(is_image(Path(p)) for p in paths):
            return self.load(paths)
        self._cannot_open("choose one video file, one folder, or several image files (not a mix, and one video at a time)")

    def open_folder(self, folder: Path) -> None:
        """A recording folder. With several videos in it, ask which one; with only images, read those."""
        from celldform.demo.source import is_image, videos_in
        vids = videos_in(folder)
        if len(vids) > 1 and self.source_box.currentData() == "video":
            names = [v.name for v in vids]
            name, ok = QtWidgets.QInputDialog.getItem(self, "Several videos in this folder", "Open which video?", names, 0, False)
            if not ok:
                return
            return self.load(str(folder / name))
        if not vids and (any(is_image(x) for x in folder.iterdir()) or any(
                x.is_dir() and x.name.lower().startswith("stills") for x in folder.iterdir())):
            return self.load(str(folder), source="images")     # a folder with no video: its pictures are the recording
        self.load(str(folder))

    def _cannot_open(self, why: str) -> None:
        self.verdict_lbl.setText(f"Cannot open: {why}")
        self.statusBar().showMessage(f"Cannot open: {why}")

    def _set_source(self, source: str) -> None:
        self.source_box.blockSignals(True)
        self.source_box.setCurrentIndex(self.source_box.findData(source))
        self.source_box.blockSignals(False)

    # -- the required files -----------------------------------------------------------------------------------------
    @property
    def models_ready(self) -> bool:
        return not M.missing_roles(self.model_checks)

    def _refresh_models_ui(self) -> None:
        for role, row in self.rows.items():
            row.set_check(self.model_checks.get(role))
        missing = M.missing_roles(self.model_checks)
        if missing:
            self.setup_lbl.setText("<b>Load first:</b> " + ", ".join(missing) + ". Nothing can be analyzed without them "
                                   "(browsing a recording works regardless).")
            self.setup_lbl.setStyleSheet("color:#b85c00;")
            self.models_box.setStyleSheet("QGroupBox{border:2px solid #e0a030;margin-top:8px;}QGroupBox::title{color:#b85c00;}")
        else:
            self.setup_lbl.setText("All required files are loaded.")
            self.setup_lbl.setStyleSheet("color:#2e7d32;")
            self.models_box.setStyleSheet("")
        self._refresh_controls()

    def _warn(self, title: str, text: str) -> None:
        self.statusBar().showMessage(text)
        if self.isVisible():
            QtWidgets.QMessageBox.warning(self, title, text)

    def load_role(self, role: str, path, remember: bool = True, quiet: bool = False) -> bool:
        """Check one file and, if it is good, use it. A file that fails is refused with the reason and whatever was
        loaded before stays in use. Returns whether the file was accepted."""
        c = M.check(role, path)
        if not c.ok:
            if not quiet:
                self._warn(M.ROLES[role].title, f"{M.ROLES[role].title}: {c.message}")
            elif role not in self.model_checks:
                self.model_checks[role] = c                    # a remembered file that has gone: show why, keep the row
            self._refresh_models_ui()
            return False
        if role == "classifier":
            from celldform.demo import framework as fw
            try:
                self.framework = fw.load(path)
            except Exception as e:                             # checked a moment ago; a file that changes meanwhile
                self._warn("HER2 classifier", str(e))
                return False
        self.model_checks[role], self.model_paths[role] = c, Path(path)
        if role in ("unet", "unet_config"):
            self.segmenter = None                              # rebuilt from the new pair below
        if remember:
            self.settings.set_model(role, path, c.sha256)
        self.settings.last_dir = str(Path(path).parent) if remember else self.settings.last_dir
        if role == "unet" and "unet_config" not in self.model_paths:       # the settings file saved beside the weights
            for name in ("unet_settings.yaml", "config.yaml"):
                if (Path(path).parent / name).is_file() and self.load_role("unet_config", Path(path).parent / name, remember, quiet=True):
                    break
        self._try_unet_pair()
        self._refresh_models_ui()
        if not quiet:
            self.statusBar().showMessage(f"{M.ROLES[role].title}: {c.message}")
        if role == "reference" and self.recording is not None:
            self.load(self.recording)
        return True

    def _try_unet_pair(self) -> None:
        """When the weights and their settings are both loaded, run the pair once on a blank frame."""
        if not ("unet" in self.model_paths and "unet_config" in self.model_paths) or self.segmenter is not None:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            seg, chk = M.try_segmenter(self.model_paths["unet"], self.model_paths["unet_config"])
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if seg is None:
            self.model_checks["unet"] = chk                    # not ok: Analyze stays off and the row says why
            self._warn("U-Net", chk.message)
            return
        self.segmenter = seg
        self.segmenter_name = f"U-Net ({self.model_paths['unet'].name})"
        self.model_checks["unet"] = chk

    def choose_role(self, role: str) -> None:
        r = M.ROLES[role]
        pattern = " ".join("*" + x for x in r.suffixes)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, r.title, self.settings.last_dir, f"{r.title} ({pattern});;All files (*)")
        if path:
            self.load_role(role, path)

    def choose_bundle_zip(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Model bundle", self.settings.last_dir, "Model bundle (*.zip)")
        if path:
            self.load_bundle(path)

    def choose_bundle_folder(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Model bundle folder", self.settings.last_dir)
        if path:
            self.load_bundle(path)

    def load_bundle(self, path) -> bool:
        """Load every file of a bundle (folder or .zip) after its checksums have been verified."""
        try:
            files = M.open_bundle(path)
        except M.BundleError as e:
            self._warn("Model bundle", str(e))
            return False
        ok = all(self.load_role(role, f, quiet=True) for role, f in files.items())
        if not ok:
            self._warn("Model bundle", "some files in the bundle are not usable: see the Required files box")
        else:
            note = M.describe_bundle(path)
            self.statusBar().showMessage("Model bundle loaded and verified" + (f": {note}" if note else "."))
        return ok

    def clear_models(self) -> None:
        self.model_checks.clear(); self.model_paths.clear()
        self.segmenter = self.framework = None
        self.segmenter_name = "U-Net"
        self.settings.forget_models()
        self._refresh_models_ui()
        if self.recording is not None:
            self.load(self.recording)

    def restore_models(self) -> None:
        """Reload the files remembered from last time; one that has gone says so instead of failing."""
        for role in M.ROLES:
            p = self.settings.model_paths().get(role)
            if p is not None:
                self.load_role(role, p, remember=False, quiet=True)
        self._refresh_models_ui()

    def source_changed(self) -> None:
        if self.recording is not None and self._run_state == "idle":
            self.load(self.recording)

    def load(self, spec, source: Optional[str] = None) -> None:
        """Open a recording: a folder, a video file (str), or a list of image files. ``source`` overrides the box."""
        from celldform.demo.source import Recording
        images_only = isinstance(spec, (list, tuple))
        if images_only:
            src = "images"
        elif source:
            src = source
        elif self._images_only:                                # the last recording was hand-picked images: start afresh
            src = "video"
        else:
            src = self.source_box.currentData()
        if self._run_state != "idle":
            self.statusBar().showMessage("Stop the analysis before opening another recording.")
            return
        try:
            rec = Recording.from_spec(spec, source=src)
        except Exception as e:                                 # whatever was loaded before stays loaded and usable
            self._cannot_open(str(e))
            self._set_source(self._source)
            self._refresh_controls()
            return
        try:
            self.player.set_recording(rec)                     # also shows frame 0
        except Exception as e:
            self._cannot_open(str(e))
            self._set_source(self._source)
            self._refresh_controls()
            return
        self.recording, self.trap = spec, None
        self._images_only, self._source = images_only, src
        self._set_source(src)
        self.total = rec.declared_frames or 0
        self.bar.setValue(0)
        self.verdict_lbl.setText("Ready")
        what = f"{len(rec.images)} still images" if rec.source == "images" else "video"
        self.info.setText(f"{rec.cell_label or rec.name}: {rec.declared_frames} frames, {rec.fps:g} fps "
                          f"({what}; {rec.name})")
        self._refresh_controls()
        self.statusBar().showMessage("Loaded. Look through it with the Browse bar, or press Analyze to run the models.")
        log = rec.frame_log                                   # before the run: the protocol and the model card
        self.dash.set_state(dashboard.State(
            t_s=log.time_s.to_numpy(float) if log is not None and "set_current_mA" in log else None,
            mA=log.set_current_mA.to_numpy(float) if log is not None and "set_current_mA" in log else None,
            fps=rec.fps, total_frames=self.total, title=str(rec.cell_label or rec.name),
            info=dict(getattr(self.framework, "info", {}) or {}),
            reference=dashboard.load_reference(self.model_paths.get("reference")) if self.model_paths.get("reference") else None))

    def pick_trap(self) -> None:
        self.picking = True
        self.info.setText("Click on the trapped cell in the image.")

    def on_click(self, x: float, y: float) -> None:
        if self.picking:
            iw, ih = self.view.image_size()                    # the trap is kept in the analysis frame (1228x922)
            self.picking, self.trap = False, (x * FEATURE_SIZE[0] / iw, y * FEATURE_SIZE[1] / ih)
            self.info.setText(f"Trap set at ({self.trap[0]:.0f}, {self.trap[1]:.0f}).")

    def on_browse_frame(self, frame: np.ndarray, idx: int) -> None:
        self.view.show_image(to_qimage(frame))

    # -- running -------------------------------------------------------------------------------------------------
    def start(self) -> None:
        """Analyze: run the models over the loaded recording."""
        if self._run_state != "idle":
            self.statusBar().showMessage("An analysis is already running.")
            return
        if not self.loaded:
            self.statusBar().showMessage("Open a recording first (Open recording menu, or drop it on the window).")
            return
        if not self.models_ready:
            self.statusBar().showMessage("Load the required files first: " + ", ".join(M.missing_roles(self.model_checks)) + ".")
            return
        try:
            self.out = self._new_results_folder()
            if self.worker is not None:                        # never replace a thread that has not quite ended
                self.worker.wait(3000)
                self._retired = [w for w in self._retired + [self.worker] if w.isRunning()]
            self.dash.log.clear()
            self.worker = w = Worker(self.recording, self.segmenter, self.framework, self.trap, self.out,
                                     self.segmenter_name, self._source, str(self.model_paths.get("unet") or "") or None,
                                     str(self.model_paths.get("unet_config") or "") or None,
                                     self.chk_provisional.isChecked(),
                                     str(self.model_paths.get("reference") or "") or None)
            w.status.connect(self.statusBar().showMessage)
            w.log_ready.connect(self._pull_log)
            w.frame_ready.connect(self._pull_frame)
            w.snapshot_ready.connect(self._pull_snapshot)
            w.failed.connect(lambda m: self.verdict_lbl.setText(m))
            w.finished.connect(lambda w=w: self.on_done(w))
            self.verdict_lbl.setText("Analyzing...")
            self._set_run_state("running")
            w.start()
        except Exception as e:
            self._set_run_state("idle")
            self.report_error("Analyze", e)

    def _new_results_folder(self) -> Path:
        """A folder of its own for this analysis, named after the recording and the time (never overwrites a previous one)."""
        name = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in Path(str(self.recording[0] if isinstance(self.recording, (list, tuple)) else self.recording)).stem)[:40]
        base = self._results_base / f"{time.strftime('%Y%m%d_%H%M%S')}_{name or 'recording'}"
        out, k = base, 1
        while out.exists():
            k += 1
            out = base.with_name(f"{base.name}_{k}")
        return out

    def toggle_pause(self) -> None:
        if self._run_state == "running":
            self.worker.pause()                               # takes effect before the next frame is read
            self._set_run_state("paused")
            self.statusBar().showMessage("Paused. Zoom and pan still work; press Resume to continue.")
        elif self._run_state == "paused":
            self.worker.resume()
            self._set_run_state("running")
            self.statusBar().showMessage("Resumed")

    def stop(self) -> None:
        if self._run_state in ("running", "paused"):
            self.worker.cancel()
            self._set_run_state("stopping")
            self.statusBar().showMessage("Stopping...")

    # -- results from the worker (newest only; see Latest) -----------------------------------------------------------
    def _pull_frame(self) -> None:
        item = self.worker.frames.take() if self.worker else None
        if item is not None:
            self.on_frame(*item)

    def _pull_snapshot(self) -> None:
        state = self.worker.snaps.take() if self.worker else None
        if state is not None:
            self.dash.set_state(state)

    def _pull_log(self) -> None:
        lines = self.worker.logs.take() if self.worker else []
        if lines:
            self.dash.append_log("\n".join(lines))

    def on_frame(self, img: QtGui.QImage, v, i: int = 0) -> None:
        if self.total:
            self.bar.setValue(min(100, int(100 * i / self.total)))
            if not (self.worker and self.worker.paused):
                self.statusBar().showMessage(f"frame {i} / {self.total}")
        self.view.show_image(img)
        self.show_verdict(v)

    BASE_STYLE = "font-size:20px;font-weight:bold;"

    def show_verdict(self, v, final: bool = False) -> None:
        self.verdict_lbl.setStyleSheet(self.BASE_STYLE + ("color:#b85c00;" if v.status == "provisional" else ""))
        if v.status == "ok":
            self.verdict_lbl.setText(f"{v.label.upper()} HER2  (P high {v.p_high:.2f})")
        elif v.status == "provisional":
            self.verdict_lbl.setText(f"PROVISIONAL, not validated: {v.label.upper()} HER2  (P high {v.p_high:.2f})")
        elif final:
            self.verdict_lbl.setText("Finished: no HER2 verdict could be given")
        else:
            self.verdict_lbl.setText("No verdict yet")
        lead = {"ok": "", "provisional": "Not a HER2 result, because: "}.get(v.status, "")
        self.info.setText((lead + v.reason + "\n" if v.status != "ok" else "") +
                          f"{v.n_steps} laser steps, {v.pass_fraction:.0%} of measured frames pass the checks")
        self.caveats.setText("\n".join("• " + c for c in v.caveats))

    def on_done(self, worker=None) -> None:
        if worker is not None and worker is not self.worker:        # an older thread ending: nothing to update
            return
        self._pull_log(); self._pull_snapshot(); self._pull_frame()          # whatever the worker left behind
        self._set_run_state("idle")
        if self.worker and self.worker.segmenter is not None:
            self.segmenter = self.worker.segmenter            # kept: a second run does not reload the model
        stopped = bool(self.worker and self.worker._stop)
        if not stopped:
            self.bar.setValue(100)
        v = self.worker.verdict if self.worker else None
        if v is not None:
            self.show_verdict(v, final=not stopped)
        elif self.worker is not None and "Analyzing" in self.verdict_lbl.text():
            self.verdict_lbl.setText("Stopped" if stopped else "Ready")
        self.statusBar().showMessage(("Stopped. " if stopped else f"Finished reading the {self._source}. ") +
                                     f"Results saved in {self.out}")

    def closeEvent(self, e) -> None:  # noqa: N802
        for w in [self.worker, *self._retired]:
            if w is not None and w.isRunning():
                w.cancel(); w.wait(5000)
        super().closeEvent(e)


def initial_size(avail_w: int, avail_h: int) -> Tuple[int, int]:
    """A first window size that fits the screen (it used to open taller than a 1080p screen)."""
    return min(int(avail_w * 0.94), 1700), min(int(avail_h * 0.9), 1100)


def fix_qt_plugin_path() -> bool:
    """``import cv2`` points Qt at OpenCV's own (Qt5) plugins, which can stop Qt from creating a window. Undo that."""
    import os
    path = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "")
    if "cv2" in path.replace("\\", "/").split("/"):
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
        if "cv2" in os.environ.get("QT_QPA_FONTDIR", "").replace("\\", "/").split("/"):
            os.environ.pop("QT_QPA_FONTDIR", None)
        return True
    return False


def app_version() -> str:
    try:
        from celldform import __version__
        return str(__version__)
    except Exception:
        return "unknown"


def selftest() -> int:
    """Build the window, draw it once, and quit: what the installer's build runs on the finished program to prove it
    starts (libraries, plugins and icon all present) without a screen, models or any file."""
    privacy.block_network()
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1200, 800)
    w.show()
    for _ in range(20):
        app.processEvents()
    ok = w.view is not None and not w.models_ready and not w.btn_analyze.isEnabled() and privacy.network_blocked()
    print(f"celldform {app_version()} selftest {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def main() -> None:
    import traceback
    if "--version" in sys.argv:
        print(f"celldform {app_version()}")
        return
    if "--selftest" in sys.argv:
        fix_qt_plugin_path()
        sys.exit(selftest())
    privacy.block_network()                                    # the app works offline; nothing leaves this machine
    fix_qt_plugin_path()
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("celldform")
    icon = Path(__file__).resolve().parent / "assets" / "celldform.png"
    if icon.is_file():
        app.setWindowIcon(QtGui.QIcon(str(icon)))
    settings = Settings.default()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    w = MainWindow(recording=args[0] if args else None, settings=settings,
                   results_dir=settings.results_dir)
    w.restore_models()                                         # the files loaded last time, checked again

    def hook(etype, value, tb):                                # PyQt aborts the program on an unhandled slot error
        traceback.print_exception(etype, value, tb)
        w.report_error("Unexpected error", value)
    sys.excepthook = hook
    avail = app.primaryScreen().availableGeometry()
    w.resize(*initial_size(avail.width(), avail.height()))
    w.show()
    if not w.models_ready:
        w.statusBar().showMessage("First load the required files (box on the right): the model bundle, or the U-Net, "
                                  "its settings and the classifier one by one.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

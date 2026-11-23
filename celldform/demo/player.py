"""
Browse a recording without analysing it: play the video or step through its still images, jump to any frame, change
the speed. The frames are shown as recorded (colour, overlay text included); nothing is segmented or measured.

:class:`PlayerBar` owns the controls and the clock; it hands each frame to whoever listens to ``frame_shown`` (the
window's image view, which does the zooming and panning). Playback follows the clock, not the frame count: when decoding
cannot keep up, frames are skipped and the video still runs at the chosen speed.
"""
from __future__ import annotations

from typing import Optional

from celldform.demo.qt import QtCore, QtWidgets

from celldform.demo.source import FrameReader, Recording

SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
STEPS = ((1, "1 frame"), (10, "10 frames"), (50, "50 frames"), (200, "1 laser step (200 frames)"), (1000, "1000 frames"))


class PlayerBar(QtWidgets.QGroupBox):
    frame_shown = QtCore.Signal(object, int)          # BGR frame, frame index
    fit_requested = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__("Browse the recording (no analysis)")
        self.rec: Optional[Recording] = None
        self.reader: Optional[FrameReader] = None
        self.idx, self.playing = 0, False
        self._clock, self._i0 = QtCore.QElapsedTimer(), 0
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(15)
        self.timer.timeout.connect(self._tick)

        mk = QtWidgets.QPushButton
        self.btn_first, self.btn_back, self.btn_play = mk("|<"), mk("<"), mk("Play")
        self.btn_fwd, self.btn_last, self.btn_fit = mk(">"), mk(">|"), mk("Fit view")
        for b, tip in ((self.btn_first, "First frame (Home)"), (self.btn_back, "Back by the step size (Left)"),
                       (self.btn_play, "Play / pause the browsing (Space)"), (self.btn_fwd, "Forward by the step size (Right)"),
                       (self.btn_last, "Last frame (End)"),
                       (self.btn_fit, "Show the whole image again (also: double-click the image). "
                                      "Zoom with the mouse wheel, pan by dragging.")):
            b.setToolTip(tip)
        for b in (self.btn_first, self.btn_back, self.btn_fwd, self.btn_last):
            b.setFixedWidth(40)
        self.btn_play.setMinimumWidth(70)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.info = QtWidgets.QLabel("no recording")
        self.speed = QtWidgets.QComboBox()
        for v in SPEEDS:
            self.speed.addItem(f"{v:g}x", v)
        self.speed.setCurrentIndex(SPEEDS.index(1.0))
        self.step_box = QtWidgets.QComboBox()
        for n, label in STEPS:
            self.step_box.addItem(label, n)
        self.step_box.setToolTip("How far < and > move: one frame, or a whole laser step to sample through the recording")
        self.no_text = QtWidgets.QCheckBox("Hide overlay text")

        row1 = QtWidgets.QHBoxLayout()
        for w in (self.btn_first, self.btn_back, self.btn_play, self.btn_fwd, self.btn_last):
            row1.addWidget(w)
        row1.addWidget(self.slider, 1)
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(self.info, 1)
        for text, w in (("speed", self.speed), ("step", self.step_box)):
            row2.addWidget(QtWidgets.QLabel(text))
            row2.addWidget(w)
        row2.addWidget(self.no_text)
        row2.addWidget(self.btn_fit)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(row1)
        lay.addLayout(row2)

        self.btn_first.clicked.connect(lambda: self.show_frame(0))
        self.btn_last.clicked.connect(lambda: self.show_frame(self.count - 1))
        self.btn_back.clicked.connect(lambda: self.step(-1))
        self.btn_fwd.clicked.connect(lambda: self.step(1))
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_fit.clicked.connect(self.fit_requested)
        self.slider.valueChanged.connect(self._slider_moved)
        self.speed.currentIndexChanged.connect(lambda _: self._reanchor())
        self.no_text.toggled.connect(self._overlay_toggled)
        self.setEnabled(False)

    # -- recording ----------------------------------------------------------------------------------------------
    @property
    def count(self) -> int:
        return self.reader.count if self.reader else 0

    def set_recording(self, rec: Recording) -> None:
        self.pause_play()
        if self.reader:
            self.reader.close()
        self.rec, self.reader, self.idx = rec, FrameReader(rec, remove_text=self.no_text.isChecked()), 0
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(self.count - 1, 0))
        self.slider.blockSignals(False)
        self.setEnabled(True)
        self.show_frame(0)

    def set_analysis_running(self, running: bool) -> None:
        """While the models run the view shows the analysed frames, so browsing is off."""
        if running:
            self.pause_play()
        self.setEnabled(bool(self.reader) and not running)

    # -- showing frames -----------------------------------------------------------------------------------------
    def show_frame(self, i: int) -> bool:
        if not self.reader:
            return False
        i = min(max(int(i), 0), self.count - 1)
        f = self.reader.read(i)
        if f is None:
            self.info.setText(f"frame {i} cannot be decoded (a damaged part of the file?)")
            self.pause_play()
            return False
        self.idx = i
        self.slider.blockSignals(True)
        self.slider.setValue(i)
        self.slider.blockSignals(False)
        self.info.setText(self._describe(i))
        self.frame_shown.emit(f, i)
        return True

    def _describe(self, i: int) -> str:
        rec = self.rec
        text = f"frame {i} / {self.count - 1}    {i / rec.fps:.1f} s"
        try:
            from celldform.analysis_config import load
            text += f"    laser step {i // load('features').frames_per_step}"
        except Exception:
            pass
        log = rec.frame_log
        if log is not None and "set_current_mA" in log and i < len(log):
            text += f"    {float(log.set_current_mA.iloc[i]):.0f} mA"
        return text

    def step(self, direction: int) -> None:
        if self.reader and self.isEnabled():
            self.show_frame(self.idx + direction * int(self.step_box.currentData()))
            self._reanchor()

    def _slider_moved(self, v: int) -> None:
        self.show_frame(v)
        self._reanchor()

    def _overlay_toggled(self, on: bool) -> None:
        if self.reader:
            self.reader.remove_text = on
            self.show_frame(self.idx)

    # -- playing ------------------------------------------------------------------------------------------------
    def _reanchor(self) -> None:
        self._i0 = self.idx
        self._clock.restart()

    def toggle_play(self) -> None:
        if not (self.reader and self.isEnabled()):
            return
        self.pause_play() if self.playing else self.play()

    def play(self) -> None:
        if not self.reader:
            return
        if self.idx >= self.count - 1:
            self.show_frame(0)
        self.playing = True
        self.btn_play.setText("Pause")
        self._reanchor()
        self.timer.start()

    def pause_play(self) -> None:
        self.playing = False
        self.timer.stop()
        self.btn_play.setText("Play")

    def _tick(self) -> None:
        if not (self.playing and self.reader):
            return
        target = int(self._i0 + self._clock.elapsed() / 1000.0 * self.rec.fps * float(self.speed.currentData()))
        if target <= self.idx:
            return
        if target >= self.count - 1:
            self.show_frame(self.count - 1)
            self.pause_play()
            return
        self.show_frame(target)

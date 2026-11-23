"""
Which Qt binding the app uses, decided in one place.

The app is written for **PySide6** (Qt for Python, LGPL: the licence that lets the program be shipped without releasing
its source under the GPL). If PySide6 is not installed but PyQt6 is, PyQt6 is used instead so that a developer's
existing environment keeps working; PyQt6 is GPL-licensed, so an installer must never be built with it (the Windows build
installs PySide6 only). ``CELLDFORM_QT=pyside6`` or ``CELLDFORM_QT=pyqt6`` forces one.

    from celldform.demo.qt import QtCore, QtGui, QtWidgets, Signal, BINDING

The rest of the code uses the PySide6 names (``Signal``); with PyQt6 they are aliased here. matplotlib is told to use
the same binding, so a window never mixes two.
"""
from __future__ import annotations

import importlib
import os

_WANT = os.environ.get("CELLDFORM_QT", "").strip().lower()


def _load():
    errors = []
    for name in (("pyqt6",) if _WANT == "pyqt6" else ("pyside6",) if _WANT == "pyside6" else ("pyside6", "pyqt6")):
        pkg = "PySide6" if name == "pyside6" else "PyQt6"
        try:
            core = importlib.import_module(f"{pkg}.QtCore")
            gui = importlib.import_module(f"{pkg}.QtGui")
            widgets = importlib.import_module(f"{pkg}.QtWidgets")
        except ImportError as e:
            errors.append(f"{pkg}: {e}")
            continue
        try:
            test = importlib.import_module(f"{pkg}.QtTest")
        except ImportError:
            test = None
        return name, core, gui, widgets, test
    raise ImportError("celldform's window needs PySide6 (pip install PySide6). Tried: " + "; ".join(errors))


BINDING, QtCore, QtGui, QtWidgets, QtTest = _load()
if BINDING == "pyqt6":                                         # one set of names in the rest of the code
    QtCore.Signal = QtCore.pyqtSignal
    QtCore.Slot = QtCore.pyqtSlot
Signal = QtCore.Signal
os.environ["QT_API"] = BINDING                                  # matplotlib's Qt backend follows the same binding

__all__ = ["BINDING", "QtCore", "QtGui", "QtWidgets", "QtTest", "Signal"]

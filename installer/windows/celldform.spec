# -*- mode: python ; coding: utf-8 -*-
# PyInstaller recipe for celldform.exe (a folder build, windowed, no console). Run by build.ps1 from a neutral folder.
from pathlib import Path

HERE = Path(SPECPATH).resolve()            # noqa: F821  (SPECPATH is supplied by PyInstaller)
SRC = HERE.parent.parent                                           # the staged copy of the repository
ICON = SRC / "celldform" / "demo" / "assets" / "celldform.ico"

datas = [
    (str(SRC / "celldform" / "demo" / "assets"), "celldform/demo/assets"),   # the icon (read at run time)
    (str(SRC / "LICENSE"), "."),
]
notices = SRC / "THIRD_PARTY_NOTICES.txt"
if notices.is_file():
    datas.append((str(notices), "."))

# imports done inside functions are found by PyInstaller too; these are listed so a mistake cannot drop one silently
hiddenimports = [
    "celldform.demo.models", "celldform.demo.privacy", "celldform.demo.settings", "celldform.demo.shortcut",
    "celldform.demo.dashboard", "celldform.demo.player", "celldform.demo.segment", "celldform.demo.framework",
    "celldform.demo.run", "celldform.demo.session", "celldform.demo.source", "celldform.demo.analyze",
    "sklearn.pipeline", "sklearn.impute._base", "sklearn.preprocessing._data", "sklearn.decomposition._pca",
    "sklearn.linear_model._logistic", "joblib.numpy_pickle", "matplotlib.backends.backend_qtagg",
]

# research-only or unneeded: the app starts without every one of these (tests/test_installer.py). PyQt5/PyQt6 are
# excluded on purpose: they are GPL, PySide6 (LGPL) is the window library. Keep this a FOLDER build (COLLECT): the Qt
# libraries stay separate files the user could replace, which is what the LGPL asks of an app that links them.
excludes = [
    "napari", "nnunetv2", "statsmodels", "torchvision", "fastapi", "uvicorn", "mpi4py", "mkdocs", "SimpleITK",
    "batchgenerators", "dynamic_network_architectures", "requests", "urllib3", "tkinter", "IPython", "jupyter",
    "notebook", "pytest", "PyQt5", "PyQt6", "PySide2", "matplotlib.tests", "numpy.tests", "scipy.tests",
]

a = Analysis(                                                      # noqa: F821
    [str(HERE / "app_entry.py")], pathex=[str(SRC)], binaries=[], datas=datas, hiddenimports=hiddenimports,
    hookspath=[], runtime_hooks=[], excludes=excludes, noarchive=False,
)
pyz = PYZ(a.pure)                                                  # noqa: F821
exe = EXE(                                                         # noqa: F821
    pyz, a.scripts, [], exclude_binaries=True, name="celldform", icon=str(ICON),
    console=False,                                                 # a window program: no black console behind it
    upx=False,                                                     # UPX-packed programs trigger antivirus warnings
    uac_admin=False,                                               # installs and runs without administrator rights
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="celldform")   # noqa: F821

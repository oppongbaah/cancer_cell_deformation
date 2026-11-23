# Building and shipping the Windows app

For whoever prepares a release. End users read [docs/install_windows.md](../docs/install_windows.md).

## What is here

| File | Purpose |
|---|---|
| `windows/build.ps1` | the whole build: neutral staging copy, audit, venv, PyInstaller, self-test, audit, Inno Setup, SHA-256 |
| `windows/celldform.spec` | PyInstaller recipe (windowed, no UPX, no admin, research-only libraries excluded) |
| `windows/celldform.iss` | Inno Setup script: per-user install, desktop shortcut task, Start Menu, clean uninstall |
| `windows/requirements-app.txt` | the lean run-time libraries (PySide6, headless OpenCV, no nnU-Net, statsmodels, napari ...) |
| `audit_package.py` | refuses to ship the maker's data (see below) |
| `third_party_notices.py` | licence notices of everything bundled |
| `make_icon.py` | draws `celldform/demo/assets/celldform.{png,ico}` |

## Build (Windows, once Python 3.10-3.12 and Inno Setup 7 (or 6) are installed)

```powershell
powershell -File installer\windows\build.ps1            # CPU build of PyTorch, about 500 MB installer
powershell -File installer\windows\build.ps1 -Gpu       # CUDA build, several GB larger
```

Output: `dist\installer\celldform-setup-<version>.exe` and `.sha256`. The same recipe runs on a clean machine in
`.github/workflows/windows-installer.yml` (manual or on a `v*` tag); it keeps the installer as a workflow artifact and
publishes nothing. **This recipe was written and its parts tested on Linux; it has not been run on Windows yet.** Do a
first build and install on a clean Windows machine (a VM) before handing it to anyone.

## Making sure no sensitive data goes out

1. **The build runs in `C:\celldform_build`**, a copy without `__pycache__`/`*.pyc`: compiled Python files remember the
   folder they were built in, and a build inside a home folder would put the user name into the program.
2. **`audit_package.py` runs three times** (the source, the frozen program, the installer file) and stops the build on:
   the maker's e-mail / GitHub name / user name / home folder anywhere; any e-mail, token, private key or user-home path
   in our own files; model, video, table, log and secret files; `data/`, `results/`, `checkpoints/`, `.git`. Third-party
   files are only searched for the maker's identifiers, so a library's own authors are not flagged.
3. **Package metadata.** `pyproject.toml` lists an author e-mail address. The installer does not carry it (it is built
   from source, without metadata), but a wheel or sdist built with `python -m build` does: audit it before uploading
   (`python installer/audit_package.py dist/celldform-*.whl`) and decide whether that address belongs there.
4. **The author's name** appears in `LICENSE` and the package docstring as attribution; the audit allows it only there.
5. **No models, data or settings ship.** Users load their own files; see the model bundle in `docs/demo.md`.
6. **No network.** The program blocks itself from reaching other computers at start; the installer needs none.

## Before shipping outside the lab

- **Licence.** The window library is **PySide6** (LGPL v3), so the installer can be shipped without releasing celldform's source
  under the GPL. The LGPL asks that the Qt libraries stay replaceable: keep the PyInstaller recipe a **folder** build (not
  one single `.exe`), so the Qt and PySide6 files remain separate files, and ship `THIRD_PARTY_NOTICES.txt` (the build does).
  **Never put PyQt6 in the installer's requirements** (it is GPL); `tests/test_installer.py` fails if it appears.
- **Code signing.** An unsigned installer triggers Windows SmartScreen. A code-signing certificate (or Azure Trusted
  Signing) removes the warning; sign `celldform.exe` and the installer after the build. Until then give out the SHA-256.
- **Version.** One place: `version` in `pyproject.toml` (also the default in `celldform.iss`; a test keeps them equal).
- **A clean-machine test.** Install, open a recording, load a bundle, analyze, uninstall; check `Documents\celldform`
  and `%APPDATA%\celldform` contain only what `docs/privacy.md` says.

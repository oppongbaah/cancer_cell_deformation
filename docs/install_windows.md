# Installing the desktop app

celldform's desktop app opens a recording of a trapped cell, runs the models and shows the HER2 class with the evidence behind
it. It works **offline** and sends nothing anywhere ([Privacy](privacy.md)).

## Do I need to install the library first?

**No, not for the Windows installer.** The installer is self-contained: Python and every library the app needs are inside it,
so you do **not** need to install Python, `pip` or the `celldform` library beforehand. Just run the installer.

You only need Python and the library if you choose the second route below.

| | **A. Windows installer** | **B. Python (`pip`)** |
|---|---|---|
| Who it is for | someone who just wants to analyze recordings | someone who also wants the library, scripts or other systems |
| What you must install first | **nothing** | Python 3.10, 3.11 or 3.12 |
| How | run `celldform-setup-<version>.exe` | `pip install "celldform[app]"` |
| How you start it | the desktop shortcut or Start Menu | `celldform-demo` (and `celldform-shortcut` for a desktop icon) |
| Size | large (it contains PyTorch) | the libraries are downloaded by `pip` |

**Both routes need the same thing before the app can analyze anything: the model files** (see [First start](#first-start-load-the-files)).
They are not part of either install.

## A. The Windows installer

1. Run `celldform-setup-<version>.exe`. It installs for you only, so **no administrator rights are needed**.
2. On the "Select Additional Tasks" page leave **Create a desktop shortcut** ticked (and tick **Add to the Start Menu** if you
   like), then install. Start celldform from the shortcut.
3. If Windows SmartScreen says *"Windows protected your PC"*, the installer is not code-signed. Check that the SHA-256 of the
   file you downloaded equals the one you were given (`Get-FileHash celldform-setup-<version>.exe` in PowerShell), then choose
   *More info → Run anyway*.

For several computers at once: `celldform-setup-<version>.exe /SILENT /TASKS="desktopicon"`.

## B. Python (`pip`)

Needs Python 3.10, 3.11 or 3.12. In a terminal:

```bash
python -m venv celldform-env
celldform-env\Scripts\activate           # Linux/macOS: source celldform-env/bin/activate
pip install "celldform[app]"             # the library and the app (PySide6)
celldform-demo                           # starts the app
celldform-shortcut                       # optional: a desktop shortcut (--start-menu, --remove)
```

Until the package is on PyPI, install from a clone of the repository instead: `pip install -e ".[app]"` inside it.
`pip install celldform` alone (without `[app]`) gives the library without the window.

## First start: load the files

The app contains **no models**. The right-hand panel, **Required files**, lists what it needs, each marked ○ (not loaded):

| File | What it is |
|---|---|
| U-Net weights (`.pt`) **and** U-Net settings (`.yaml`) | the segmentation model, and how frames are prepared for it |
| *or* nnU-Net checkpoint (`.pth`) | the segmentation model, as one file (the more accurate, slower one) |
| HER2 classifier (`.joblib`) | the fitted classifier |

You need one segmentation model (the U-Net pair, or the nnU-Net; both can be loaded and you choose) plus the classifier. Until then,
**Analyze** stays grey and its tooltip says what is missing. You can already open a recording and
browse it (play, step, zoom).

**Where do the files come from?** From whoever trained the models; the project does not distribute them. Ask for a **model
bundle**, one `.zip` that holds them all.

Easiest: **Load a model bundle ▾ → Bundle .zip file…** and pick the `.zip`. Every file inside is checked against its checksum and
the marks turn ✔. Or load the files one by one with each row's **Load…** button (the settings file is found by itself when it
sits next to the weights). A wrong file is refused with the reason, and what was loaded before stays loaded.

The app remembers which files you loaded and checks them again at the next start, so this is a one-time step. **Clear all**
forgets them (the files themselves are never deleted or changed).

## Use

1. **Open recording ▾**: a video file, several image files, or a recording folder (or drop them on the window).
2. Browse with the bar under the picture; zoom with the mouse wheel, pan by dragging, **F11** for full screen.
3. **Analyze**, then **Pause / Resume / Stop** as needed. The dashboard and the Log tab fill in as it runs.
4. **Open results folder** (or `Documents\celldform\results`): each analysis writes a folder named after the recording and the
   time, with `verdict.json`, `frames.csv`, `steps.csv`, `dashboard.png` and `inference.log`. They do not say which folder the
   recording came from or who ran it.

The app gives a verdict only when a recording has at least 3 usable laser steps (about 27 s) and 80 % of the measured frames pass the checks;
otherwise it says why. From 10 steps (about 91 s) it uses the full validated classifier; shorter recordings get a result labelled
as a short recording, from the step model alone. Details, and the optional provisional verdict, are in [the demonstration app](demo.md).

## Uninstall

*Windows installer:* Settings → Apps → celldform → Uninstall. It asks whether to also delete the saved settings (the list of model
files); your results in `Documents\celldform` are never deleted, and neither are the model files you loaded.
*pip:* `pip uninstall celldform`, then delete `celldform-env` and, if you made one, the desktop shortcut (`celldform-shortcut --remove`).

## Troubleshooting

| What you see | What it means |
|---|---|
| Analyze is grey | A recording is not open, or a required file is not loaded: the tooltip says which. |
| ✘ next to a file | The reason is shown under its name: wrong kind of file, damaged, settings that do not fit, or a classifier trained on other features. |
| "does not match its SHA-256" | The bundle changed or was damaged in transit: copy or download it again. |
| "not part of a HER2 classifier; the file was not opened" | The `.joblib` contains something a classifier never contains. Do not use it. |
| "the U-Net and its settings do not work together" | The `.pt` and `.yaml` come from different models, or the graphics driver failed. |
| `celldform-demo` is not found | The environment is not active (run the `activate` line again), or you installed without `[app]`. |
| No window opens | Run `celldform-demo --selftest` (or `celldform.exe --selftest` for the installer); it prints what failed. |

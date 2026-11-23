# Demonstration application

Runs the trained models on a newly recorded video and reports the HER2 class of the trapped SKBR3 cell. **Not
real-time** (a measured frame costs a network pass; the 45.5 ms camera interval is only a speed reference).

```bash
celldform-demo [recording]                          # the window (Windows installer: the desktop shortcut)
python -m celldform.demo.run <recording> --out <dir> --bundle models.zip      # headless
python -m celldform.demo.models make --unet w.pt --config c.yaml --classifier f.joblib --out models   # prepare the files
```

**The app contains no models.** On a new machine it cannot analyze anything until the required files are loaded; see
[Required files](#required-files-first-run-on-a-new-machine) below, and [Installing on Windows](install_windows.md).

A recording folder holds `video.avi` and, if present, `SKBR3.txt` and `frames.csv` (per-frame laser log).

## What it does

1. `source.py` reads the video (a damaged tail ends the stream, it never raises), removes the burned-in overlay text,
   converts to gray 1228x922.
2. Only the measured frames are decoded and segmented: in every 200-frame laser step, offsets 0, 20, 60, 100, 140, 180
   (`configs/analysis/features.yaml`).
3. `segment.py`: the U-Net you loaded (its crop and filters from its settings file) -> labels at full frame.
4. `analyze.py`: trapped cell = region nearest the trap (learned from the frames seen so far, or set by a click), the
   five frame checks, shape and deformation features.
5. `session.py`: laser steps accumulate; the framework classifier (step LR + LR on PC1 of the selected 4, equal-weight
   mean, 14/15 leave-one-cell-out) gives P(high) **only** with >= 10 laser steps and >= 80 % of measured frames passing.
   Otherwise the verdict is withheld with the reason.

Outputs of the headless run: `annotated.mp4` (predicted mask outlined white, decoy grey), `frames.csv`, `steps.csv`,
`verdict.json` (includes the caveats and timing).

## Caveats the app prints

- Masks come from the U-Net you loaded; the classifier's features were measured on nnU-Net masks.
- A laser current range or hold time different from the training recordings (500-1200 mA) is reported from `frames.csv`.
- The crop centre of the U-Net is a calibration of the training setup's trap; a recording from another setup needs
  `--crop-center`.

## Known limitation

The app works on recordings made like the ones the models were trained on. On four recordings from a newer setup (colour
camera, a 298-1000 mA ramp in steps of about 9 s instead of 500-1200 mA held for 30 s) the U-Net found the cell in only 2-13 %
of the measured frames, so the app correctly gave no verdict; a different pre-processing or a model fine-tuned on frames of
that setup is needed there. The dashboard shows why frames were rejected, which tells you whether a recording is the
problem. See [Project status](status.md).

## Required files (first run on a new machine)

Nothing is bundled with the program, so a new installation has to be given a **segmentation model** and the **HER2
classifier** before **Analyze** works (browsing a recording needs none). The segmentation model is *either* the U-Net
weights with their settings file, *or* an nnU-Net checkpoint (or both). The **Required files** box lists the files with a mark
each: ✔ loaded and checked, ✘ the file was refused (with the reason), ○ not loaded yet.

| File | What it is | What is checked |
|---|---|---|
| **U-Net weights** (`.pt`) | the segmentation model | it is a U-Net with 2 or 3 classes (width and classes are read from the weights); read with `weights_only`, so it cannot run code |
| **U-Net settings** (`.yaml`) | filters, crop size and crop centre the model was trained with | every value is present, sensible and fits a 1228x922 frame; found automatically when `unet_settings.yaml` or `config.yaml` sits beside the weights |
| **nnU-Net checkpoint** (`.pth`), instead of the two above | one nnU-Net 2d checkpoint; nothing else is needed | read safely (only tensors, plain values and NumPy numbers are accepted, so it cannot run code); a 2d model with one input channel and labels 0 = background, 1 = trapped cell; then run once on a blank frame |
| **HER2 classifier** (`.joblib`) | the fitted step model + cell model | it contains only what a HER2 framework contains (everything else is refused unopened), and it expects the features this version measures |
| Training-cell reference (`.csv`, optional) | the 15 training cells' features | has the columns the comparison chart needs; without it that chart says so |

**Which model runs.** When both a U-Net and an nnU-Net are loaded, **Segment with** chooses; the nnU-Net is the default
because it is the model the classifier's features were measured on (so the "masks are not from nnU-Net" caveat disappears).
It is more accurate (per-cell Dice 0.94 against 0.87 for the U-Net in leave-one-cell-out testing) but slower: about 0.5 s a
frame on a GPU and several seconds on a CPU, so a recording takes minutes instead of under a minute. A recording of a few
hundred measured frames is fine on a GPU; on a PC without one the U-Net is the practical choice.

**The nnU-Net file.** nnU-Net normally wants a whole model folder; everything it needs is inside its `.pth`, so the app
reads that one file and writes a stripped copy (without the optimizer state and training log, half the size, same
predictions) into its cache folder the first time. It needs the `nnunetv2` package, which the Windows installer includes
and `pip install "celldform[app]"` installs; without it the row says so and the U-Net still works.

When the weights and their settings are both loaded they are run together once on a blank frame, which shows (and
proves) that they work on this machine and which device (CPU or GPU) they use. The files you loaded are remembered
(their paths and SHA-256 only) and checked again at the next start; one that has gone or changed is shown as ✘ rather
than failing.

**Model bundle.** One folder or `.zip` that carries all of it, with a SHA-256 for every file, so a file damaged or swapped on
the way is refused and nothing is half-loaded. Use **Load a model bundle** in the box. Whoever holds the trained files
makes it once:

```bash
python -m celldform.demo.models make --unet unet_deploy.pt --config config.yaml --classifier her2_framework.joblib \
       [--nnunet checkpoint_final.pth] [--reference features_cells.csv] --description "U-Net + 15-cell framework" \
       --out models --zip models.zip
python -m celldform.demo.models verify models.zip
```

A bundle is rebuilt, not copied: the U-Net is re-saved as bare weights (no optimizer state, about a third of the size),
the settings file is rebuilt from the few keys the app reads (a training `config.yaml` holds dataset paths and split
details that do not belong on someone else's computer), and an nnU-Net checkpoint is re-saved without its optimizer state
and training log (474 MB becomes about 240 MB). A bundle may carry the nnU-Net alone, the U-Net pair alone, or both. The description and the classifier's own record are
refused if they contain an e-mail address or a path. A `.zip` is unpacked into the user's cache folder with protection
against entries that try to write elsewhere, links, and zip bombs.

A `.joblib` file is a pickle, and a pickle can run code when it is opened. The loader therefore only accepts the ten
classes a HER2 framework is made of and refuses anything else without opening it; `celldform.demo.framework.load(path,
trusted=True)` is plain `joblib.load` for a file you made yourself.

Command line: `--bundle models.zip`, or `--unet-file w.pt --unet-config c.yaml --classifier f.joblib`, or
`--nnunet-file model.pth --classifier f.joblib`; with both models given, `--use unet|nnunet` chooses (default nnunet). Developers on a
machine that has the repository's checkpoints can still use `--segmenter unet|nnunet`.

## Dashboard

Under the video the window has three tabs (`celldform/demo/dashboard.py`); they fill in while the recording runs.

| Tab | Panel | What it shows |
|---|---|---|
| Process | Laser protocol and what was measured | set current over the video against the 500-1200 mA the classifier was trained on; each 200-frame step coloured green / amber / red / grey |
| Process | Frames passing the checks, per laser step | pass rate per step against the 80% the classifier needs |
| Process | Why frames were rejected | no cell near the trap, too far, uncertain boundary, size off typical, ... |
| Evidence | Cell shape across the laser ramp | median area and aspect ratio of passing frames per step against the current |
| Evidence | Where this cell sits among the 15 training cells | the cell on two of the classifier's selected features, with the training cells coloured by HER2 class |
| Evidence | The cell across laser steps | a picture of the cell with its predicted outline, one per step |
| Result and model | Result | HER2 class with a probability gauge (step model, cell model, average); without a verdict, what is still missing |
| Result and model | Run and model | segmentation model and device, frames measured, time per frame (camera interval 45.5 ms is a reference only), classifier training record |

**Save dashboard image** (toolbar) writes every page on one PNG; every run also writes `dashboard.png` into its
results folder, including from the command line. Before a run the tabs already show the recording's laser protocol
and the model card.


## Controls, frame source and log

- **Analyze / Pause (Resume) / Stop** are buttons at the top of the right-hand panel; **Analyze** is what runs the
  models (the Browse bar only looks). Pause holds the run before the next frame is read; Stop works while paused and keeps the results so far. Paused time is left out
  of the timings (`verdict.json` has `paused_seconds`).
- **Frames from: Video | Images (stills)** in the toolbar (`--source video|images` on the command line). Images are the
  `stills*/` folder of the recording, or any folder of images. With `stills.csv` each picture is placed at its video
  frame (`video_frame` is 1-based), so laser steps, times and the laser log are the same as for the video; frames that
  have no still are skipped, and unselected frames are never read, which is faster than decoding the video. Without the
  table the sorted file order is the frame index. The stills carry the same camera image and the same burned-in overlay
  as the video, so they do not segment any better (checked 2026-11-18).
- **Log** is the fourth dashboard tab: model load, the settings and caveats, one line per measured frame (frame, laser
  step, segmentation time, PASS with area and aspect ratio or REJECT with the reason), a line whenever another laser
  step becomes usable (with what the verdict is then), pauses, and the result. The same lines are saved as
  `inference.log` in the results folder, also from the command line.


## Browsing a recording (no analysis)

The **Browse the recording** bar under the picture plays the video or steps through its still images (whichever
**Frames from** says) without running any model.

| Control | What it does |
|---|---|
| Play / Pause, space | plays at the chosen speed (0.25x to 16x); follows the clock, so frames are skipped rather than slowed when decoding falls behind |
| slider | jump to any frame |
| `|<` `<` `>` `>|`, Home / Left / Right / End | first frame, back, forward, last frame |
| step | how far `<` and `>` move: 1, 10, 50 frames, **1 laser step (200 frames)**, or 1000 frames, to sample through a recording |
| Hide overlay text | inpaints the burned-in laser readout (the analysis always does) |

The line under the slider gives the frame, time, laser step and set current of the frame on show. Browsing is switched
off while a run is going, and back on when it ends.

**Zoom and pan** work on the picture at any time, also during a run and while paused: mouse wheel zooms at the cursor
(up to 32x, real pixels from 3x), drag pans, double-click or **Fit view** shows the whole picture. The status bar shows
the pixel under the cursor (x, y, gray). A trap marked with **Click the cell** is converted to analysis-frame pixels
whatever the zoom or the size of the picture.

## Pause

Pause holds the run before the next frame is read (not only after a measured one), the picture shows **PAUSED** and the
status bar says Paused; zoom and pan still work. The window receives only the newest frame, dashboard snapshot and
log lines in a batch, so a slow screen cannot build up a backlog of frames that keeps playing after Pause (that is
what made Pause look broken: with a window slower than the models, the video ran on for seconds).


## Opening a recording

**Open recording** (toolbar menu, or drop on the window) takes three kinds of input:

- **Video file...**: that one file, also when its folder holds many videos. A folder with one video behaves as before
  (`frames.csv` = laser log, the `*.txt` = notes). With several videos in a folder, a video only picks up sidecars named
  after it: `<video name>.csv` (laser log) and `<video name>.txt` (notes); otherwise there is no laser log and the app
  says so in its caveats.
- **Image files... (select several)**: exactly the pictures you choose. If they come from a `stills*/` folder whose
  `stills.csv` lists them, each keeps its video frame number (so laser steps and the laser log line up, and a subset
  such as frames 600-1200 is analysed as steps 3-5); otherwise they are numbered in name order (`img2` before `img10`).
  **Frames from** is then fixed to Images. The laser log is found in the recording folder when it holds one video.
- **Recording folder...**: as before; with several videos in it the window asks which one to open, and a folder with
  only pictures is read as images.

Command line: `python -m celldform.demo.run <folder | video.avi | img1.png img2.png ...> --out <dir>`.


## Stages: what can be pressed when

One function (`MainWindow._refresh_controls`) decides this from two facts, whether a recording is loaded and what the
analysis is doing, and runs after every change, so no stage can be left with a stale button.

| Stage | Analyze | Pause | Stop | Browse bar, Open, Models, Frames from |
|---|---|---|---|---|
| nothing loaded | off (pressing it explains why) | off | off | Open only |
| loaded, idle (also after playing the video) | **on** | off | off | on |
| analyzing | off | **on** | **on** | off |
| paused | off | **on, reads Resume** | **on** | off |
| stopping (until the thread has ended) | off | off | off | off |

- A failed open (wrong file, no stills) leaves the recording that was loaded usable; the Frames from box goes back
  to what it was.
- A recording cannot be opened or changed while an analysis is running or paused (Stop first).
- An error inside the analysis, or while starting it, returns the window to idle with Analyze enabled and the
  error in the status bar and the Log tab. Unexpected errors in the window do not close it.
- `tests/test_demo_states.py` walks every stage, including random sessions of button presses with the invariants
  checked after each press.
- `import cv2` makes Qt look for OpenCV's own plugins, which can stop Qt from creating a window; `main()` undoes it.


## Provisional verdicts (when validation fails)

By default the app **withholds** a verdict unless the recording has at least 10 usable laser steps and at least 80% of
measured frames pass the five frame checks, because those are the conditions the classifier was validated under
(leave-one-cell-out 14/15, AUC 0.94, on cells that passed them).

The **Give a provisional verdict even if the checks fail** box (command line: `--provisional`) also computes a verdict
from whatever steps did pass. It is opt-in, off by default, locked while a run is going, and everything it produces
says so:

- the window shows *PROVISIONAL, not validated: HIGH HER2 (P high 0.52)* in orange, with *Not a HER2 result, because:*
  and every reason (too few steps, too low a pass rate, cell-level measures that were missing and replaced by the
  training cells' median);
- the dashboard result panel carries a **PROVISIONAL: NOT VALIDATED** badge and the reasons, and the saved dashboard
  image starts with the same words;
- `verdict.json` has `"status": "provisional"` and `"validated": false`; the log and `inference.log` say
  `PROVISIONAL, NOT VALIDATED`;
- with no usable step at all (nothing passed) there is still no verdict, and a recording that does pass validation
  gets an ordinary verdict whatever the box says.

The 14/15 figure does not apply to a provisional verdict, and neither does any claim about accuracy. Example: on
a recording of a low-HER2 cell where only 13 % of the measured frames passed the checks, the provisional verdict was *high*
with P = 0.52 (step model 0.54, cell model 0.51): a coin flip that happened to land on the wrong side. Do not read anything
into such a number.

## Window size, Maximize and full screen

The window opens at a size that fits the screen. When it is wide enough (maximized, or full screen) it switches to a wide
arrangement: the picture takes the whole height on the left and the controls and the dashboard share the right, so a big
screen is used instead of showing black bars; smaller windows keep the stacked arrangement. The switch keeps everything
(frame, tab, log, zoom). Maximize keeps the system's title bar, as on any window; **F11** goes to true full screen
(F11 or Esc to leave).

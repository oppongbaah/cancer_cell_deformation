# Privacy: what stays on your computer

**celldform sends nothing anywhere.** It has no account, no update check, no analytics and no crash reporting, and it
works with the network cable unplugged.

## How that is enforced

- At start the program switches off network access for itself: any attempt to reach another computer raises an error,
  even from a library it uses (connections to the same machine, which Qt and Python itself use, stay allowed). Telemetry
  switches of the libraries it contains are set off. `tests/test_demo_privacy.py` checks this, and that a full analysis
  works with the network blocked.
- The installer installs the program only: no models, no data, no background service, nothing that starts with Windows.
- The build refuses to produce an installer that carries its maker's data: `installer/audit_package.py` searches
  everything that would ship for the maker's e-mail address, GitHub name and user name, the build computer's home folder
  (compiled Python files remember it, so the build runs in a neutral folder), access tokens, private keys, and data files
  (models, checkpoints, videos, tables, results, logs).

## What the program writes, and where

| What | Where | Contains |
|---|---|---|
| Settings | `%APPDATA%\celldform\settings.json` | the paths and SHA-256 of the model files you loaded, the last folder, nothing else |
| Results | `Documents\celldform\results\<time>_<recording>\` | the analysis of one recording: verdict, per-frame and per-step tables, dashboard picture, log |
| Unpacked bundles | `%LOCALAPPDATA%\celldform\Cache\bundles` | the files of a `.zip` bundle you loaded |

Result files are written so they can be shared: the recording is named by its file name only, the log never contains the
folder it came from, and your home folder, user name and e-mail addresses are replaced (`~`, `<user>`, `<email>`).
Set `CELLDFORM_HOME` to a folder to keep settings and results there instead (for example on a USB stick).

## What you should still decide yourself

- The videos and the model files are yours; the program never copies them. A model bundle carries only weights, a small
  settings file and the classifier, never the training data or the training configuration.
- A `.joblib` classifier is a kind of file that can contain code. The program opens only the ten kinds of object a HER2
  classifier consists of and refuses the rest unopened, but still load model files only from a source you trust, and
  compare the SHA-256 you were given.

# Analysis settings

One YAML file per analysis stage, read by `celldform.analysis_config`. Every script takes `--config <file>` to
use a copy instead, and writes the settings it actually used next to its results (`config_*.yaml`).

| File | Script(s) | Results (and the settings used) |
|---|---|---|
| `features.yaml` | `scripts/features/extract_features.py` | `results/features/<model>/config_<step>.yaml` |
| `cnn.yaml` | `scripts/classification/frames_cnn.py` | `results/classification/<model>/config_cnn_<step>.yaml` |
| `classifiers.yaml` | `scripts/classification/benchmark_cells.py` | `results/classification/<model>/config_classifiers.yaml` |
| `regression.yaml` | `scripts/biomechanics/deformation_regression.py`, `rate_of_deformation.py` | `results/biomechanics/<model>/config_{regression,rod}.yaml` |

The values in these files are the ones the reported results were produced with; a missing file or key falls back
to the same values. A misspelt key is an error.

**Tuning with 14 cells.** Every result is scored leave-one-cell-out on the same 14 cells. Changing a setting,
rerunning and keeping the better score tunes the setting *on the test cells*, and the score is then optimistic.
To tune honestly: copy the file (`--config my.yaml`), decide what to try before looking, report every setting
tried, or tune inside a nested cross-validation and confirm on new cells.

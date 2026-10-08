# HPC Guide (NSU)

Running `celldform` U-Net training on the NSU HPC cluster with SLURM.

| | |
|---|---|
| Node | `casev-hpc-lg01` (login and compute) |
| Partitions | `standard` (default), `gpu` — both on the same node, no time limit |
| GPUs | 3× Tesla V100-PCIE-32GB (driver 580, CUDA ≤ 13) |
| CPU / RAM | 80 cores, ~500 GB |
| Account / QOS | not required |

!!! note "V100, not A100"
    The V100 has FP16 tensor cores but **no BF16 and no TF32**. `training.use_amp: true` uses
    FP16 autocast plus a `GradScaler` (`celldform/segmentation/trainer.py`), which is what the
    V100 accelerates, so leave it on. Epoch times measured earlier on other hardware do not
    carry over; time a short run before sizing `--time`.

---

## Recommended Workflow

| Step | Where |
|------|--------|
| Annotate masks | Local machine (`scripts/annotation/annotate.py`) |
| Preprocess frames | Local machine (`scripts/data/preprocess_frames.py --masks`) |
| Upload preprocessed data | `rsync`/`scp` to NSU |
| Clone code, create conda env | NSU login shell |
| Train / evaluate | NSU, `sbatch scripts/hpc/{training,evaluation}/*.sh` (V100) |
| Download checkpoint | `rsync`/`scp` from NSU |
| Inference, evaluation, biomechanics | Local machine (or NSU) |

---

## Step 1 — Clone the Code

```bash
cd ~/Desktop/Isaac/thesis            # or wherever you keep the project
git clone https://github.com/oppongbaah/cancer_cell_deformation.git
cd cancer_cell_deformation
mkdir -p logs/hpc                    # SLURM writes job logs here and will not create it
```

---

## Step 2 — Set Up the Environment

Use the cluster's Anaconda module and the `celldform` conda env (once, on the login node):

```bash
module load Anaconda3/2025.06-1
conda create -n celldform python=3.11 -y     # skip if it already exists (conda env list)
conda activate celldform
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # FIRST -- see below
pip install -e ".[hpc]"    # core deps incl. nnunetv2 (needs Python >= 3.10) + mpi4py

# Verify (the GPUs are visible on the login node too)
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nnUNetv2_train -h
```

Notes for NSU:

- Batch jobs don't run your `~/.bashrc`, so the job script loads the module itself and sources
  `$EBROOTANACONDA3/etc/profile.d/conda.sh` before running `conda activate celldform`.
- **Install torch from the `cu126` index.** The default PyPI wheel is now built for CUDA 13
  (`+cu130`), which has no kernels for the V100 (compute capability 7.0; `cu130` starts at
  7.5). It installs fine, then warns *"does not include kernels for this GPU"* and fails on the
  first CUDA op. The `cu126` build includes `sm_70`. To check:
  `python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"` must list
  `sm_70`. If a later `pip install` pulls in a `+cu130` torch again, reinstall the same version
  from `cu126`: `pip uninstall -y torch torchvision && pip install torch==<ver> torchvision==<ver>
  --index-url https://download.pytorch.org/whl/cu126`.
- Don't build the env on the `PyTorch/2.9.1-CUDA-12.6.0` module: **nnunetv2 excludes torch
  2.9.x** (`torch!=2.9.*`). Let pip choose torch; its wheels bundle the CUDA runtime, and the
  driver (580) supports it.
- Every script in `scripts/training/` and `scripts/evaluation/` is Python, and the ones that
  chain others (`train_experiment.py`, `train_cv_fold.py`, ...) run each step with whichever
  python launched them. `python scripts/...` therefore uses the conda env on NSU and `.venv`
  locally, with nothing to configure.

---

## Step 3 — Upload Preprocessed Data

From your local machine:

```bash
rsync -avP data/preprocessed/ YOUR_USERNAME@NSU_HPC_HOST:~/Desktop/Isaac/thesis/cancer_cell_deformation/data/preprocessed/
```

Only `preprocessed/` is needed. Do not upload `data/raw/`, which holds the large videos and is not used in training.

---

## Step 4 — Check Paths in the Config

Data paths in `configs/*.yaml` are relative to the repo root, so they work unchanged as long as
the job runs from there (both the batch script and `celldform-hpc-submit` do). The settings
that matter on NSU:

```yaml
device: cuda

training:
  num_workers: 8   # match --cpus-per-task; beyond 16 adds no throughput
  batch_size: 32   # a V100 has 32 GB; raise once you have >= 100 annotated masks
  use_amp: true    # FP16 autocast on V100 tensor cores
```

!!! warning "Reproducibility across runs"
    `training.cudnn_benchmark` (default `true`, matching PyTorch's own default) trades reproducibility for speed by re-timing convolution kernels against runtime hardware state. It was verified locally to make same-`seed` reruns land on different results even with `cudnn_deterministic: true` also set (see `CLAUDE.md`'s "Training non-determinism found" note). This has not been re-verified on NSU's V100s. If you need a run to be reproducible, set `cudnn_benchmark: false` explicitly.

---

## Step 5 — Submit Jobs

Every training and evaluation script has a SLURM job script under `scripts/hpc/`, mirroring
`scripts/training/` and `scripts/evaluation/`. Submit **from the repo root**. Arguments after
the script name are passed straight to the underlying script:

```bash
sbatch scripts/hpc/training/train_unet.sh                                       # configs/default.yaml
sbatch scripts/hpc/training/train_unet.sh --config configs/multiclass_experiment_256.yaml --seed 43
sbatch scripts/hpc/training/train_experiment.sh postswap/experiment_1b_loss e1b_lovasz_s42
sbatch --array=1-5 scripts/hpc/training/train_cv_fold.sh randswap/experiment_0_cv_baseline cv_baseline
sbatch scripts/hpc/evaluation/eval_native.sh --checkpoint CK --config CFG --log results/<exp>/eval_log_native.jsonl
sbatch scripts/hpc/evaluation/summarize_cv.sh --exp randswap/experiment_0_cv_baseline --run cv_baseline
```

| Job script | Runs | Partition | Resources (default) |
|---|---|---|---|
| `training/train_unet.sh` | `scripts/training/train_unet.py` (no args → `--config configs/default.yaml`) | gpu | 1 V100, 8 CPU, 32 GB, 12 h |
| `training/distill_ensemble.sh` | `scripts/training/distill_ensemble.py` | gpu | 1 V100, 8 CPU, 32 GB, 12 h |
| `training/train_experiment.sh` | `scripts/training/train_experiment.py <experiment> <run> [train args]`: train + score | gpu | 1 V100, 8 CPU, 32 GB, 12 h |
| `training/train_cv_fold.sh` | `scripts/training/train_cv_fold.py <experiment> <run> <fold>`; under `--array` the fold is the array index | gpu | 1 V100, 8 CPU, 32 GB, 12 h |
| `training/train_deploy_model.sh` | `scripts/training/train_deploy_model.py <experiment> <run>` | gpu | 1 V100, 8 CPU, 32 GB, 12 h |
| `training/train_nnunet_cv.sh` | `scripts/training/train_nnunet_cv.py` (`--setup` once first) | gpu | 1 V100, 16 CPU, 64 GB, 72 h |
| `training/train_student_cv.sh` | `scripts/training/train_student_cv.py` | gpu | 1 V100, 8 CPU, 32 GB, 48 h |
| `evaluation/evaluate_unet.sh` | `scripts/evaluation/evaluate_unet.py` | gpu | 1 V100, 8 CPU, 32 GB, 2 h |
| `evaluation/eval_native.sh` | `scripts/evaluation/eval_native.py` | gpu | 1 V100, 8 CPU, 32 GB, 2 h |
| `evaluation/eval_testing_set.sh` | `scripts/evaluation/eval_testing_set.py <ckpt> <config> <log> [pool]` | gpu | 1 V100, 8 CPU, 32 GB, 2 h |
| `evaluation/ensemble_eval.sh` | `scripts/evaluation/ensemble_eval.py` | gpu | 1 V100, 8 CPU, 32 GB, 2 h |
| `evaluation/eval_ensemble_cv.sh` | `scripts/evaluation/eval_ensemble_cv.py` | gpu | 1 V100, 8 CPU, 64 GB, 8 h |
| `evaluation/latency_benchmark.sh` | `scripts/evaluation/latency_benchmark.py` (add `--exclusive` for publishable timings) | gpu | 1 V100, 8 CPU, 32 GB, 2 h |
| `evaluation/eval_nnunet.sh` | `scripts/evaluation/eval_nnunet.py` | standard | 4 CPU, 16 GB, 1 h |
| `evaluation/summarize_cv.sh` | `scripts/evaluation/summarize_cv.py` | standard | 4 CPU, 16 GB, 30 min |
| `evaluation/compare_postswap.sh` | `scripts/evaluation/compare_postswap.py` | standard | 4 CPU, 16 GB, 30 min |
| `evaluation/check_convergence.sh` | `scripts/evaluation/check_convergence.py` | standard | 4 CPU, 16 GB, 30 min |

The old `run_postswap.sh <run>` is gone. Use `train_experiment.sh postswap/experiment_0a_seed_baseline <run>`.

Override resources at submit time without editing a file, e.g. `sbatch --time=24:00:00 --mem=64G ...`.
`sbatch --test-only <job script>` validates the request without submitting. Every job sources
`scripts/hpc/_common.sh`, which activates the `celldform` conda env (`--export=ALL,CONDA_ENV=<name>`
for another) and prints the job's GPU and torch build.

**Logs.** Each job writes `logs/hpc/<job-name>_<jobid>.out` (stdout and stderr together). A
matching **`.err` file exists only when the job failed**: non-zero exit, time limit or
`scancel`. It holds the job's stderr plus a closing line giving the exit code or the
SIGTERM. Successful jobs leave no `.err`, so `ls logs/hpc/*.err` lists the failures.

**Auto-commit.** When a job exits 0, `scripts/hpc/_autocommit.sh` commits its outputs and pushes
them to the current branch on GitHub. A failed, cancelled or timed-out job commits nothing.

- *What is committed:* that job's `logs/hpc/<job>_<jobid>.out`, plus `results/` and `logs/` as far
  as `.gitignore` admits them (eval logs `*.jsonl`, training logs `*.log`). Figures, CSVs and
  checkpoints stay out. **Code is never committed by a job**, and anything else you have staged or
  edited is left as it is (`git commit --only`).
- *Where:* the current branch, never `main`/`master` (on those it does nothing). Jobs finishing
  together commit one at a time (a lock in `.git/`).
- *How it pushes:* over SSH with the repo's deploy key `~/.ssh/celldform_deploy` (GitHub → repo
  Settings → Deploy keys, write access). Your own remote and credentials are not used. If a
  push fails, the commit stays local and the job still succeeds; push it yourself.
- *Record:* one line per job in `logs/hpc/autocommit.log` (not tracked), e.g.
  `committed 55a2b28 on feat-hpc and pushed`.
- *Switches:* `sbatch --export=ALL,AUTOCOMMIT=0 ...` skips it; `AUTOCOMMIT_PUSH=0` commits without pushing.

`celldform-hpc-submit` generates the `train_unet` job on the fly and submits it (same setup, via `_common.sh`):

```bash
celldform-hpc-submit --config configs/default.yaml
celldform-hpc-submit --config configs/default.yaml --time 24:00:00 --dry-run   # print, don't submit
```

Defaults: `--partition gpu --time 12:00:00 --cpus 8 --gpus 1 --mem 32G --conda-env celldform`, no account or QOS (pass `--account`/`--qos` only if NSU starts requiring them).

Monitoring (SLURM accounting is disabled on NSU, so `sacct` returns nothing; use the logs):

```bash
squeue -u $USER                                   # queued/running status
squeue -u $USER --start                           # estimated start time, if SLURM has computed one
scontrol show job <jobid>                         # details while it is queued/running
tail -f logs/hpc/<job-name>_<jobid>.out           # live output; the last line gives the exit code
ls logs/hpc/*.err                                 # which jobs failed
scancel <jobid>                                   # cancel
```

The node has 3 V100s, so up to three single-GPU jobs (e.g. three seeds or CV folds) run side by side.

---

## Step 6 — Download the Checkpoint

From your local machine:

```bash
rsync -avP YOUR_USERNAME@NSU_HPC_HOST:~/Desktop/Isaac/thesis/cancer_cell_deformation/checkpoints/ checkpoints/
```

---

## Iterative Training Cycle

```
Local                              NSU
─────────────────────              ──────────────────────────────────
scripts/annotation/annotate.py
scripts/data/preprocess_frames.py
                      rsync data/preprocessed/ ──►
                                  sbatch scripts/hpc/training/train_unet.sh
                      ◄── rsync checkpoints/
scripts/evaluation/evaluate_unet.py
```

Repeat after each annotation batch. Target DSC ≥ 0.80 before freezing the model.

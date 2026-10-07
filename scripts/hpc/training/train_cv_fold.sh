#!/bin/bash
#SBATCH --job-name=train_cv_fold
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1        # one of the node's 3x Tesla V100-PCIE-32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/hpc/%x_%j.out
#
# One randswap cross-validation round: build the fold, train, score natively
# (scripts/training/train_cv_fold.py). Submit once per fold; the 3 GPUs run 3 folds at once.
#
# Submit from the repo root. Arguments after the script name are passed through:
#   sbatch scripts/hpc/training/train_cv_fold.sh randswap/experiment_0_cv_baseline cv_baseline 1
#   sbatch --array=1-5 scripts/hpc/training/train_cv_fold.sh randswap/experiment_0_cv_baseline cv_baseline   # fold = array index
#
# Override resources at submit time, e.g. sbatch --time=24:00:00 --mem=64G ...
# Logs: logs/hpc/train_cv_fold_<jobid>.out. A .err file there exists only if the job failed
# (see scripts/hpc/_common.sh).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/hpc/_common.sh

# Under --array, the fold is the array index unless given explicitly.
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ $# -eq 2 ] && set -- "$@" "$SLURM_ARRAY_TASK_ID"
run_py scripts/training/train_cv_fold.py "$@"

#!/bin/bash
#SBATCH --job-name=distill_ensemble
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1        # one of the node's 3x Tesla V100-PCIE-32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/hpc/%x_%j.out
#
# Distil the 4-seed ensemble into one U-Net (scripts/training/distill_ensemble.py): caches the
# teachers' soft targets, then trains the student.
#
# Submit from the repo root. Arguments after the script name are passed through:
#   sbatch scripts/hpc/training/distill_ensemble.sh
#   sbatch scripts/hpc/training/distill_ensemble.sh --cache-only
#   sbatch scripts/hpc/training/distill_ensemble.sh --temperature 3 --lambda-hard 0.5
#
# Override resources at submit time, e.g. sbatch --time=24:00:00 --mem=64G ...
# Logs: logs/hpc/distill_ensemble_<jobid>.out. A .err file there exists only if the job failed
# (see scripts/hpc/_common.sh).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/hpc/_common.sh

run_py scripts/training/distill_ensemble.py "$@"

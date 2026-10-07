#!/bin/bash
#SBATCH --job-name=train_nnunet_cv
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1        # one of the node's 3x Tesla V100-PCIE-32GB
#SBATCH --time=72:00:00
#SBATCH --output=logs/hpc/%x_%j.out
#
# nnU-Net benchmark over the randswap rounds: train + predict + score
# (scripts/training/train_nnunet_cv.py). Run --setup once first.
#
# Submit from the repo root. Arguments after the script name are passed through:
#   sbatch scripts/hpc/training/train_nnunet_cv.sh --setup
#   sbatch scripts/hpc/training/train_nnunet_cv.sh
#   sbatch scripts/hpc/training/train_nnunet_cv.sh --rounds 3 4
#   sbatch scripts/hpc/training/train_nnunet_cv.sh --rounds 2 --continue-training
#
# Override resources at submit time, e.g. sbatch --time=24:00:00 --mem=64G ...
# Logs: logs/hpc/train_nnunet_cv_<jobid>.out. A .err file there exists only if the job failed
# (see scripts/hpc/_common.sh).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/hpc/_common.sh

run_py scripts/training/train_nnunet_cv.py "$@"

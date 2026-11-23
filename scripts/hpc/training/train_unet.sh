#!/bin/bash
#SBATCH --job-name=train_unet
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1        # one of the node's 3x Tesla V100-PCIE-32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/hpc/%x_%j.out
#
# Train the U-Net (scripts/training/train_unet.py).
#
# Submit from the repo root. Arguments after the script name are passed through:
#   sbatch scripts/hpc/training/train_unet.sh
#   sbatch scripts/hpc/training/train_unet.sh --config configs/multiclass_experiment_256.yaml
#   sbatch scripts/hpc/training/train_unet.sh --config configs/default.yaml --seed 43 --tag auto
#
# Override resources at submit time, e.g. sbatch --time=24:00:00 --mem=64G ...
# Logs: logs/hpc/train_unet_<jobid>.out. A .err file there exists only if the job failed
# (see scripts/hpc/_common.sh).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/hpc/_common.sh

[ $# -gt 0 ] || set -- --config configs/default.yaml
run_py scripts/training/train_unet.py "$@"

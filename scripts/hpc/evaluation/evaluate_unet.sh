#!/bin/bash
#SBATCH --job-name=evaluate_unet
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1        # one of the node's 3x Tesla V100-PCIE-32GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/hpc/%x_%j.out
#
# Score a checkpoint on a holdout pool, at its own size (scripts/evaluation/evaluate_unet.py).
#
# Submit from the repo root. Arguments after the script name are passed through:
#   sbatch scripts/hpc/evaluation/evaluate_unet.sh --checkpoint checkpoints/unet_best.pt --pool 02_unet_holdout
#
# Override resources at submit time, e.g. sbatch --time=24:00:00 --mem=64G ...
# Logs: logs/hpc/evaluate_unet_<jobid>.out. A .err file there exists only if the job failed
# (see scripts/hpc/_common.sh).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/hpc/_common.sh

run_py scripts/evaluation/evaluate_unet.py "$@"

#!/bin/bash
#SBATCH --job-name=train_deploy_model
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1        # one of the node's 3x Tesla V100-PCIE-32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/hpc/%x_%j.out
#
# The randswap deployment model: every cell trains, for the CV median best epoch
# (scripts/training/train_deploy_model.py). Run after the CV rounds + summarize_cv.
#
# Submit from the repo root. Arguments after the script name are passed through:
#   sbatch scripts/hpc/training/train_deploy_model.sh randswap/experiment_0_cv_baseline cv_baseline
#
# Override resources at submit time, e.g. sbatch --time=24:00:00 --mem=64G ...
# Logs: logs/hpc/train_deploy_model_<jobid>.out. A .err file there exists only if the job failed
# (see scripts/hpc/_common.sh).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/hpc/_common.sh

run_py scripts/training/train_deploy_model.py "$@"

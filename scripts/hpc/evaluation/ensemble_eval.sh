#!/bin/bash
#SBATCH --job-name=ensemble_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1        # one of the node's 3x Tesla V100-PCIE-32GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/hpc/%x_%j.out
#
# Evaluate an ensemble of checkpoints on the development set (scripts/evaluation/ensemble_eval.py).
#
# Submit from the repo root. Arguments after the script name are passed through:
#   sbatch scripts/hpc/evaluation/ensemble_eval.sh
#   sbatch scripts/hpc/evaluation/ensemble_eval.sh --runs A B C
#   sbatch scripts/hpc/evaluation/ensemble_eval.sh --save-log
#
# Override resources at submit time, e.g. sbatch --time=24:00:00 --mem=64G ...
# Logs: logs/hpc/ensemble_eval_<jobid>.out. A .err file there exists only if the job failed
# (see scripts/hpc/_common.sh).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/hpc/_common.sh

run_py scripts/evaluation/ensemble_eval.py "$@"

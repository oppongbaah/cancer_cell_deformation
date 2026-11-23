#!/bin/bash
#SBATCH --job-name=eval_native
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1        # one of the node's 3x Tesla V100-PCIE-32GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/hpc/%x_%j.out
#
# Score a checkpoint at native 1228x922 resolution (scripts/evaluation/eval_native.py).
#
# Submit from the repo root. Arguments after the script name are passed through:
#   sbatch scripts/hpc/evaluation/eval_native.sh --checkpoint CK --config CFG --inputs data/preprocessed/evaluate/postswap_devset_512/inputs --log results/postswap/<experiment>/eval_log_native.jsonl
#
# Override resources at submit time, e.g. sbatch --time=24:00:00 --mem=64G ...
# Logs: logs/hpc/eval_native_<jobid>.out. A .err file there exists only if the job failed
# (see scripts/hpc/_common.sh).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/hpc/_common.sh

run_py scripts/evaluation/eval_native.py "$@"

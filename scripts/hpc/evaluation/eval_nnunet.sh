#!/bin/bash
#SBATCH --job-name=eval_nnunet
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/hpc/%x_%j.out
#
# Score one round of nnU-Net predictions at native resolution (scripts/evaluation/eval_nnunet.py).
# Compares saved label maps; no GPU needed.
#
# Submit from the repo root. Arguments after the script name are passed through:
#   sbatch scripts/hpc/evaluation/eval_nnunet.sh --round 1 --pred-dir data/nnunet/predictions/round1 --checkpoint <fold_0/checkpoint_final.pth> --log results/randswap/experiment_1_nnunet/eval_log_native.jsonl
#
# Override resources at submit time, e.g. sbatch --time=24:00:00 --mem=64G ...
# Logs: logs/hpc/eval_nnunet_<jobid>.out. A .err file there exists only if the job failed
# (see scripts/hpc/_common.sh).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/hpc/_common.sh

run_py scripts/evaluation/eval_nnunet.py "$@"

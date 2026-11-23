#!/bin/bash
#SBATCH --job-name=compare_postswap
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/hpc/%x_%j.out
#
# Compare postswap runs against the baseline, noise floor enforced (scripts/evaluation/compare_postswap.py). CPU only.
#
# Submit from the repo root. Arguments after the script name are passed through:
#   sbatch scripts/hpc/evaluation/compare_postswap.sh
#   sbatch scripts/hpc/evaluation/compare_postswap.sh --runs A B
#   sbatch scripts/hpc/evaluation/compare_postswap.sh --seeds 42 43 44
#
# Override resources at submit time, e.g. sbatch --time=24:00:00 --mem=64G ...
# Logs: logs/hpc/compare_postswap_<jobid>.out. A .err file there exists only if the job failed
# (see scripts/hpc/_common.sh).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/hpc/_common.sh

run_py scripts/evaluation/compare_postswap.py "$@"

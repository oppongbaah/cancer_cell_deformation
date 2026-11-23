#!/bin/bash
#SBATCH --job-name=summarize_cv
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/hpc/%x_%j.out
#
# Summarise a randswap cross-validation into CSV/JSON (scripts/evaluation/summarize_cv.py). CPU only.
#
# Submit from the repo root. Arguments after the script name are passed through:
#   sbatch scripts/hpc/evaluation/summarize_cv.sh --exp randswap/experiment_0_cv_baseline --run cv_baseline
#
# Override resources at submit time, e.g. sbatch --time=24:00:00 --mem=64G ...
# Logs: logs/hpc/summarize_cv_<jobid>.out. A .err file there exists only if the job failed
# (see scripts/hpc/_common.sh).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/hpc/_common.sh

run_py scripts/evaluation/summarize_cv.py "$@"

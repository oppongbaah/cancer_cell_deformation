#!/bin/bash
#SBATCH --job-name=check_convergence
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/hpc/%x_%j.out
#
# Flag runs that converged poorly, from their checkpoints' internal val Dice
# (scripts/evaluation/check_convergence.py). CPU only.
#
# Submit from the repo root. Arguments after the script name are passed through:
#   sbatch scripts/hpc/evaluation/check_convergence.sh
#   sbatch scripts/hpc/evaluation/check_convergence.sh --runs A B C
#   sbatch scripts/hpc/evaluation/check_convergence.sh --tolerance 0.015
#
# Override resources at submit time, e.g. sbatch --time=24:00:00 --mem=64G ...
# Logs: logs/hpc/check_convergence_<jobid>.out. A .err file there exists only if the job failed
# (see scripts/hpc/_common.sh).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/hpc/_common.sh

run_py scripts/evaluation/check_convergence.py "$@"

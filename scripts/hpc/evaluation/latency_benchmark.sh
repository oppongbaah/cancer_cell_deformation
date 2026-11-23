#!/bin/bash
#SBATCH --job-name=latency_benchmark
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1        # one of the node's 3x Tesla V100-PCIE-32GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/hpc/%x_%j.out
#
# Frame-to-mask latency, our U-Net vs nnU-Net (scripts/evaluation/latency_benchmark.py).
# The timings assume an idle GPU. --gres gives this job its own V100, but the other two may
# be busy. For publishable numbers, submit with --exclusive (takes the whole node).
#
# Submit from the repo root. Arguments after the script name are passed through:
#   sbatch scripts/hpc/evaluation/latency_benchmark.sh --unet U.pt --config CFG --crop-center-from SPLIT.json --nnunet NNUNET_DIR --fold 0 --out results/.../latency_ms.json
#
# Override resources at submit time, e.g. sbatch --time=24:00:00 --mem=64G ...
# Logs: logs/hpc/latency_benchmark_<jobid>.out. A .err file there exists only if the job failed
# (see scripts/hpc/_common.sh).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/hpc/_common.sh

run_py scripts/evaluation/latency_benchmark.py "$@"

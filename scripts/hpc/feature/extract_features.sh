#!/bin/bash
#SBATCH --job-name=extract_features
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1        # one of the node's 3x Tesla V100-PCIE-32GB
#SBATCH --time=06:00:00
#SBATCH --output=logs/hpc/%x_%j.out
#
# Feature extraction with the cross-validation checkpoints that live on the cluster
# (scripts/features/extract_features.py): segment every feature frame and hand-masked test frame with
# the round model that never saw its cell, validate, extract. Then pack the predictions
# (data/features/<model>/, ~12 MB of PNGs) into results/features/<model>/predictions.tar, so the
# predictions and the result tables travel back with the run's auto-commit.
#
# Submit from the repo root. Arguments after the script name are passed through:
#   sbatch scripts/hpc/evaluation/extract_features.sh all --model nnunet1000_loocv
#
# Unpack on the workstation (repo root):  tar -xf results/features/nnunet1000_loocv/predictions.tar
# Logs: logs/hpc/extract_features_<jobid>.out. A .err file there exists only if the job failed
# (see scripts/hpc/_common.sh).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/hpc/_common.sh

run_py scripts/features/extract_features.py "$@"

MODEL=nnunet1000_loocv
args=("$@")
for i in "${!args[@]}"; do [ "${args[$i]}" = "--model" ] && MODEL="${args[$((i + 1))]}"; done
if [ -d "data/features/$MODEL" ]; then
    mkdir -p "results/features/$MODEL"
    tar -cf "results/features/$MODEL/predictions.tar" --exclude='_tmp_*' "data/features/$MODEL"
    echo "[celldform] predictions packed: results/features/$MODEL/predictions.tar ($(du -h "results/features/$MODEL/predictions.tar" | cut -f1))"
fi

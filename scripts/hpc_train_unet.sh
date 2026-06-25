#!/bin/bash
#SBATCH --job-name=celldform-unet
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --output=logs/unet_%j.out
#SBATCH --error=logs/unet_%j.err

set -euo pipefail

echo "[celldform] Job started: $(date)"
echo "[celldform] Node: $SLURMD_NODENAME"
echo "[celldform] Job ID: $SLURM_JOB_ID"
echo ""

module load anaconda
conda activate celldform

# Verify GPU
python -c "import torch; print('[celldform] CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

# Train
celldform-train-unet --config configs/default.yaml

echo ""
echo "[celldform] Job finished: $(date)"

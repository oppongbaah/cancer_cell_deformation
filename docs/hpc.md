# HPC Guide

Running celldform on Anvil (Purdue ACCESS) or any SLURM cluster.

---

## Installation on Anvil

```bash
module load python/3.10
python3 -m venv $SCRATCH/celldform_env
source $SCRATCH/celldform_env/bin/activate
pip install -e ".[hpc]"
```

The `[hpc]` extra adds `mpi4py` for distributed training.

---

## Key config changes for HPC

In `configs/default.yaml`:

```yaml
device: "cuda"               # always use GPU on Anvil
training:
  num_workers: 8             # increase for fast NFS storage
  batch_size: 32             # larger batch fits in A100 VRAM
```

Use environment variable expansion for paths:

```yaml
data:
  frames_dir: "$SCRATCH/celldform/data/preprocessed/train/01_annotate_pool/inputs"
  masks_dir:  "$SCRATCH/celldform/data/preprocessed/train/01_annotate_pool/labels"
checkpoint_dir: "$SCRATCH/celldform/checkpoints"
```

---

## SLURM job script

```bash
#!/bin/bash
#SBATCH --job-name=celldform_unet
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/unet_%j.out

module load python/3.10
source $SCRATCH/celldform_env/bin/activate

python scripts/train_unet.py --config configs/default.yaml
```

Submit with:

```bash
sbatch scripts/hpc_train_unet.sh
```

---

## Transferring data to Anvil

```bash
# From local machine
rsync -avz data/preprocessed/ anvil.rcac.purdue.edu:$SCRATCH/celldform/data/preprocessed/

# Transfer checkpoints back
rsync -avz anvil.rcac.purdue.edu:$SCRATCH/celldform/checkpoints/ checkpoints/
```

---

## Monitoring jobs

```bash
squeue -u $USER                 # list running jobs
scancel <jobid>                 # cancel a job
tail -f logs/unet_<jobid>.out   # stream output
```

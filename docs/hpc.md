# HPC Guide (Anvil)

Running `celldform` U-Net training on Purdue Anvil via ACCESS.
Verified workflow: **12× speedup** over a local GPU (0.3 s/epoch on Anvil A100 vs 3.8 s/epoch locally).

---

## Prerequisites

- An ACCESS account with an Anvil allocation (or added as a sub-user by your advisor)
- Access to the [Anvil OnDemand portal](https://ondemand.anvil.rcac.purdue.edu)

---

## Recommended Workflow

The split below keeps large raw data local and only sends what the training loop actually needs to Anvil.

| Step | Where |
|------|--------|
| Annotate masks | Local machine (`scripts/annotate.py`) |
| Preprocess frames | Local machine (`scripts/preprocess_frames.py --masks`) |
| Upload preprocessed data | Anvil OnDemand file browser |
| Clone code | Anvil terminal (`git clone`) |
| Train U-Net | Anvil (A100 GPU) |
| Download checkpoint | Anvil OnDemand file browser |
| Inference, evaluation, biomechanics | Local machine |

---

## Step 1 — Start an Interactive Session

1. Log in to [ondemand.anvil.rcac.purdue.edu](https://ondemand.anvil.rcac.purdue.edu)
2. Go to **Interactive Apps → Code Server (VS Code)**
3. Request resources — recommended for U-Net training:
    - Partition: `gpu`
    - CPUs: `8`  (more than 16 gives no benefit — workers idle waiting for the GPU)
    - GPUs: `1`
    - Memory: `32 GB`
    - Time: `2–4 hours`
4. Click **Launch** and wait for the session to start, then **Connect to VS Code**

---

## Step 2 — Upload Preprocessed Data

Anvil uses Duo two-factor authentication, which blocks standard `scp`/`rsync` without extra configuration. The easiest alternative is the OnDemand file browser.

1. On the OnDemand dashboard go to **Files** in the top menu
2. Navigate to `/anvil/scratch/YOUR_USERNAME/thesis/cancer_cell_deformation/data/`
   (create the folder first if it does not exist)
3. Click **Upload** and drag in your local `data/preprocessed/` folder

Only `preprocessed/` is needed — do not upload `data/raw/` (large videos, not used in training).

!!! tip "scp with Duo"
    If you prefer the terminal, `scp` works when you append `,push` to your password:
    ```bash
    scp -r data/preprocessed/ YOUR_USERNAME@anvil.rcac.purdue.edu:/anvil/scratch/YOUR_USERNAME/thesis/cancer_cell_deformation/data/preprocessed/
    # When prompted: yourpassword,push  (triggers a Duo push to your phone)
    ```

---

## Step 3 — Clone the Code

In the VSCode terminal on Anvil:

```bash
cd $SCRATCH/thesis
git clone https://github.com/oppongbaah/cancer_cell_deformation.git
cd cancer_cell_deformation
```

---

## Step 4 — Set Up the Environment

```bash
module load anaconda
conda create -n celldform python=3.10 -y
conda activate celldform
pip install -e ".[hpc]"

# Verify
celldform-train-unet --help
```

---

## Step 5 — Configure Paths for Anvil

Edit `configs/default.yaml` to point to the uploaded data:

```yaml
device: cuda

data:
  frames_dir: "$SCRATCH/thesis/cancer_cell_deformation/data/preprocessed/train/01_annotate_pool/inputs"
  masks_dir:  "$SCRATCH/thesis/cancer_cell_deformation/data/preprocessed/train/01_annotate_pool/labels"

checkpoint_dir: "$SCRATCH/thesis/cancer_cell_deformation/checkpoints"

training:
  num_workers: 8   # 8–16 is sufficient; beyond 16 adds no throughput
  batch_size: 32   # increase to 64–128 once you have ≥ 100 annotated masks
  use_amp: true    # enables A100 BF16/TF32 — ~2–4× throughput gain
```

---

## Step 6 — Verify GPU and Train

```bash
# Confirm the A100 is visible
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Train
celldform-train-unet --config configs/default.yaml
```

Expected throughput: **~0.3 s/epoch** on a single A100.

---

## Step 7 — Download the Checkpoint

1. On the OnDemand dashboard go to **Files**
2. Navigate to `/anvil/scratch/YOUR_USERNAME/thesis/cancer_cell_deformation/checkpoints/`
3. Select `unet_best.pt` and click **Download**
4. Move it into your local `checkpoints/` folder

---

## SLURM Batch Jobs (Optional)

For unattended overnight runs instead of an interactive session:

```bash
#!/bin/bash
#SBATCH --job-name=celldform-unet
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --output=logs/unet_%j.out

module load anaconda
conda activate celldform

celldform-train-unet --config configs/default.yaml
```

```bash
mkdir -p logs
sbatch scripts/hpc_train_unet.sh

squeue -u $USER                  # monitor job status
tail -f logs/unet_<jobid>.out    # stream live output
scancel <jobid>                  # cancel if needed
```

---

## Iterative Training Cycle

```
Local                              Anvil
─────────────────────              ──────────────────────────────────
scripts/annotate.py
scripts/preprocess_frames.py
                      upload preprocessed/ via OnDemand file browser ──►
                                           celldform-train-unet
                      ◄── download unet_best.pt via OnDemand file browser
scripts/evaluate_unet.py
```

Repeat after each annotation batch. Target DSC ≥ 0.80 before freezing the model.

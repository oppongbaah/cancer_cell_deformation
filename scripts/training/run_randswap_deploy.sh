#!/usr/bin/env bash
# The randswap DEPLOYMENT model -- the one RealTimePipeline runs on the camera: every cell
# trains, fixed length = the CV median best epoch.
#
#   scripts/training/run_randswap_deploy.sh <era>/<experiment> <run>
#   e.g. scripts/training/run_randswap_deploy.sh randswap/experiment_0_cv_baseline cv_baseline
#
# The epoch count is read from results/<experiment>/<run>_cv_summary.json (summarize_cv.py),
# never typed in. No cell is held out, so there is no score here -- the CV rounds are the
# estimate of how this model does on a new cell.
#   checkpoint  <checkpoint_dir>/fold_deploy/unet_deploy.pt
#   log         logs/<experiment>/<run>_fold_deploy.log
set -euo pipefail
cd "$(dirname "$0")/../.."
EXP="${1:?experiment}"; RUN="${2:?run}"
CFG="configs/${EXP}/${RUN}.yaml"
LOG="logs/${EXP}/${RUN}_fold_deploy.log"
EPOCHS=$(.venv/bin/python -c "import json; print(int(json.load(open('results/${EXP}/${RUN}_cv_summary.json'))['median_best_epoch']))")
ROOT=$(.venv/bin/python -c "from celldform.config import load; print(load('$CFG').split.data_root)")

echo "[$EXP] deploy: building data for all cells  ($(date +%H:%M))"
.venv/bin/python scripts/data/build_randswap.py --config "$CFG" --fold deploy | tee "$LOG"
cp "${ROOT}/fold_deploy/split.json" "results/${EXP}/${RUN}_fold_deploy_split.json"

echo "[$EXP] deploy: training ${EPOCHS} epochs (CV median best epoch)  ($(date +%H:%M))"
.venv/bin/python -u scripts/training/train_unet.py --config "$CFG" --fold deploy --epochs "$EPOCHS" \
    --tag "${RUN}_fold_deploy" --results-dir "results/${EXP}" >> "$LOG" 2>&1
tail -3 "$LOG"
echo "[$EXP] deploy done  ($(date +%H:%M))"

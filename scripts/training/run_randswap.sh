#!/usr/bin/env bash
# One randswap cross-validation round: draw + build the fold's data, train, score.
#
#   scripts/training/run_randswap.sh <era>/<experiment> <run> <fold>
#   e.g. scripts/training/run_randswap.sh randswap/experiment_0_cv_baseline cv_baseline 1
#
#   config      configs/<experiment>/<run>.yaml          (must be COMMITTED -- the draw)
#   data        <split.data_root>/fold_<k>/              (+ split.json, the record)
#   checkpoint  <checkpoint_dir>/fold_<k>/unet_best.pt
#   log         logs/<experiment>/<run>_fold_<k>.log
#   results     results/<experiment>/: curves, fold_<k>_split.json, eval_log_native.jsonl
set -euo pipefail
cd "$(dirname "$0")/../.."
EXP="${1:?experiment}"; RUN="${2:?run}"; FOLD="${3:?fold}"
CFG="configs/${EXP}/${RUN}.yaml"
mkdir -p "logs/${EXP}" "results/${EXP}"
LOG="logs/${EXP}/${RUN}_fold_${FOLD}.log"

echo "[$EXP] fold $FOLD: drawing the split and building data  ($(date +%H:%M))"
.venv/bin/python scripts/data/build_randswap.py --config "$CFG" --fold "$FOLD" | tee "$LOG"
ROOT=$(.venv/bin/python -c "from celldform.config import load; print(load('$CFG').split.data_root)")
CKROOT=$(.venv/bin/python -c "from celldform.config import load; print(load('$CFG').checkpoint_dir)")
cp "${ROOT}/fold_${FOLD}/split.json" "results/${EXP}/${RUN}_fold_${FOLD}_split.json"

echo "[$EXP] fold $FOLD: training  ($(date +%H:%M))"
.venv/bin/python -u scripts/training/train_unet.py --config "$CFG" --fold "$FOLD" \
    --tag "${RUN}_fold_${FOLD}" --results-dir "results/${EXP}" >> "$LOG" 2>&1
grep -E 'randswap fold|per-cell balanced|Early stopping|Reloaded best' "$LOG" || true

echo "[$EXP] fold $FOLD: scoring the test cells at native resolution"
.venv/bin/python scripts/evaluation/eval_native.py --checkpoint "${CKROOT}/fold_${FOLD}/unet_best.pt" \
    --config "$CFG" --fold-dir "${ROOT}/fold_${FOLD}" \
    --log "results/${EXP}/eval_log_native.jsonl" 2>/dev/null | tee -a "$LOG"
echo "[$EXP] fold $FOLD done  ($(date +%H:%M))"

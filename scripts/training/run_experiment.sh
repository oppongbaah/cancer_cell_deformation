#!/usr/bin/env bash
# Train + evaluate one run of an experiment, keeping every artefact inside that
# experiment's folder (2026-09-30 layout: one folder per experiment, mirrored under
# configs/ checkpoints/ logs/ results/).
#
#   scripts/training/run_experiment.sh <experiment> <run> [extra train args...]
#   e.g. scripts/training/run_experiment.sh postswap/experiment_1b_loss e1b_lovasz_s42
#
#   config      configs/<experiment>/<run>.yaml   (its checkpoint_dir must be
#                                                  checkpoints/<experiment>/<run>)
#   log         logs/<experiment>/<run>.log
#   curves      results/<experiment>/unet_training_curves_v*_<run>.png
#   eval log    results/<experiment>/eval_log.jsonl   (per cell + all cells, testing set, own size)
#               results/<experiment>/eval_log_native.jsonl   (same, at native 1228x922)
set -euo pipefail
cd "$(dirname "$0")/../.."
EXP="${1:?usage: run_experiment.sh <experiment> <run> [extra train args]}"
RUN="${2:?usage: run_experiment.sh <experiment> <run> [extra train args]}"
shift 2
CFG="configs/${EXP}/${RUN}.yaml"
CK="checkpoints/${EXP}/${RUN}/unet_best.pt"
[ -f "$CFG" ] || { echo "no such config: $CFG"; exit 1; }
grep -q "checkpoint_dir: \"checkpoints/${EXP}/${RUN}\"" "$CFG" \
    || { echo "$CFG: checkpoint_dir must be checkpoints/${EXP}/${RUN}"; exit 1; }
mkdir -p "logs/${EXP}" "results/${EXP}"

echo "[$EXP] training $RUN  ($(date +%H:%M))"
.venv/bin/python -u scripts/training/train_unet.py --config "$CFG" --tag "$RUN" \
    --results-dir "results/${EXP}" "$@" > "logs/${EXP}/${RUN}.log" 2>&1
grep -E 'Early stopping|Reloaded best' "logs/${EXP}/${RUN}.log" || true

# Score on the testing set at the config's own input size, then at native resolution
# (one grid for every model, so runs at different sizes compare like with like).
# The testing set matching the training data: postswap_multiclass<_v> -> postswap_devset<_v>
# (_256 -> plain postswap_devset, which predates the suffixes).
VARIANT=$(.venv/bin/python -c "from celldform.config import load; d=load('$CFG').data.frames_dir; print(d.split('postswap_multiclass')[1].split('/')[0])")
POOL="postswap_devset${VARIANT}"; [ "$VARIANT" = "_256" ] && POOL=postswap_devset
[ -d "data/preprocessed/evaluate/${POOL}" ] || { echo "no testing set data/preprocessed/evaluate/${POOL}"; exit 1; }
scripts/evaluation/eval_testing_set.sh "$CK" "$CFG" "results/${EXP}/eval_log.jsonl" "$POOL"
.venv/bin/python scripts/evaluation/eval_native.py --checkpoint "$CK" --config "$CFG" \
    --inputs "data/preprocessed/evaluate/${POOL}/inputs" --log "results/${EXP}/eval_log_native.jsonl" \
    2>/dev/null | tail -2
echo "[$EXP] $RUN done  ($(date +%H:%M))"

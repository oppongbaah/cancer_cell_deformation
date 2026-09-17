#!/usr/bin/env bash
# Train + evaluate one run of the postswap series, keeping every artefact in the
# right folder. Usage:  scripts/run_postswap.sh <run-name>  [extra train args...]
#   e.g. scripts/run_postswap.sh postswap_ftl_cw142_res384_s42
# Expects configs/postswap/<run-name>.yaml to exist.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:?usage: run_postswap.sh <run-name> [extra train args]}"; shift || true
CFG="configs/postswap/${RUN}.yaml"
[ -f "$CFG" ] || { echo "no such config: $CFG"; exit 1; }
mkdir -p logs/postswap results/postswap

echo "[postswap] training $RUN"
.venv/bin/python -u scripts/train_unet.py --config "$CFG" --tag "$RUN" "$@" \
    > "logs/postswap/${RUN}.log" 2>&1
grep -E 'split assignment|val  :|train:|WARNING' "logs/postswap/${RUN}.log" || true

CK="checkpoints/postswap/${RUN}/unet_best.pt"
echo "[postswap] evaluating $RUN on the four development cells"
for c in high_cell1 high_cell9 low_cell5 low_cell6; do
    .venv/bin/python scripts/evaluate_unet.py --checkpoint "$CK" --config "$CFG" \
        --pool postswap_devset --filter "$c" \
        --results-log results/postswap/postswap_eval_log.jsonl >/dev/null
done
.venv/bin/python scripts/evaluate_unet.py --checkpoint "$CK" --config "$CFG" \
    --pool postswap_devset --results-log results/postswap/postswap_eval_log.jsonl >/dev/null
mv -f results/unet_training_curves_*"${RUN}"*.png results/postswap/ 2>/dev/null || true

.venv/bin/python scripts/compare_postswap.py --runs "$RUN" postswap_ftl_cw142_s42

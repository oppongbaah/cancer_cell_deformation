#!/usr/bin/env bash
# Score one checkpoint on the testing set (the four held-out cells, on disk
# 02_unet_holdout / preprocessed postswap_devset): each cell, then all cells pooled.
#
#   scripts/training/eval_testing_set.sh <checkpoint.pt> <config.yaml> <eval_log.jsonl> [pool]
#   pool defaults to postswap_devset (256x256); a 512 model needs postswap_devset_512.
set -euo pipefail
cd "$(dirname "$0")/../.."
CK="${1:?checkpoint}"; CFG="${2:?config}"; LOG="${3:?eval log}"; POOL="${4:-postswap_devset}"
[ -f "$CK" ] || { echo "no checkpoint: $CK"; exit 1; }
for c in high_cell1 high_cell9 low_cell5 low_cell6 ""; do
    .venv/bin/python scripts/training/evaluate_unet.py --checkpoint "$CK" --config "$CFG" \
        --pool "$POOL" ${c:+--filter "$c"} --results-log "$LOG" >/dev/null
done
echo "  scored $(basename "$(dirname "$CK")") on the testing set -> $LOG"

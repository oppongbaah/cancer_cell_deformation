#!/usr/bin/env bash
# Superseded 2026-09-30 by run_experiment.sh, which files every artefact under one
# experiment folder. Kept so old commands still work: runs <run-name> as part of the
# seed-baseline experiment (configs/postswap/experiment_0a_seed_baseline/<run-name>.yaml).
#   scripts/training/run_postswap.sh postswap_ftl_cw142_s46
set -euo pipefail
exec "$(dirname "$0")/run_experiment.sh" postswap/experiment_0a_seed_baseline "$@"

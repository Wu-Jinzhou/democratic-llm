#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

RUN_PREP="${RUN_PREP:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_GENERATE="${RUN_GENERATE:-1}"
RUN_JUDGE="${RUN_JUDGE:-1}"
RUN_SCORE="${RUN_SCORE:-1}"

log "PRISM soft-weighting run configuration:"
log "  MODEL_ID=$MODEL_ID"
log "  NUM_TRAIN_STEPS=$NUM_TRAIN_STEPS"
log "  RUN_PREP=$RUN_PREP"
log "  RUN_TRAIN=$RUN_TRAIN"
log "  RUN_GENERATE=$RUN_GENERATE"
log "  RUN_JUDGE=$RUN_JUDGE"
log "  RUN_SCORE=$RUN_SCORE"
log "  SOFT_LINEAR_DATASET=$SOFT_LINEAR_DATASET"
log "  SOFT_SQRT_DATASET=$SOFT_SQRT_DATASET"
log "  SOFT_SQUARE_DATASET=$SOFT_SQUARE_DATASET"
log "  SOFT_CLIPPED_DATASET=$SOFT_CLIPPED_DATASET"

if [[ "$RUN_PREP" == "1" ]]; then
  "$SCRIPT_DIR/prepare_datasets.sh"
else
  log "Skipping dataset preparation because RUN_PREP=$RUN_PREP."
fi

if [[ "$RUN_TRAIN" == "1" ]]; then
  "$SCRIPT_DIR/train_soft_sqrt.sh"
  "$SCRIPT_DIR/train_soft_square.sh"
  "$SCRIPT_DIR/train_soft_clipped.sh"
else
  log "Skipping training because RUN_TRAIN=$RUN_TRAIN."
fi

if [[ "$RUN_GENERATE" == "1" ]]; then
  "$SCRIPT_DIR/generate.sh"
else
  log "Skipping response generation because RUN_GENERATE=$RUN_GENERATE."
fi

if [[ "$RUN_JUDGE" == "1" ]]; then
  "$SCRIPT_DIR/judge.sh"
else
  log "Skipping judging because RUN_JUDGE=$RUN_JUDGE."
fi

if [[ "$RUN_SCORE" == "1" ]]; then
  "$SCRIPT_DIR/score.sh"
else
  log "Skipping scoring because RUN_SCORE=$RUN_SCORE."
fi

log "Finished PRISM soft-weighting run."

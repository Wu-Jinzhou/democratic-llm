#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

RUN_TRAIN="${RUN_TRAIN:-0}"
RUN_GENERATE="${RUN_GENERATE:-0}"
RUN_JUDGE="${RUN_JUDGE:-0}"
RUN_SCORE="${RUN_SCORE:-1}"

log "PRISM k-sensitivity run configuration:"
log "  MODEL_ID=$MODEL_ID"
log "  NUM_TRAIN_STEPS=$NUM_TRAIN_STEPS"
log "  RUN_TRAIN=$RUN_TRAIN"
log "  RUN_GENERATE=$RUN_GENERATE"
log "  RUN_JUDGE=$RUN_JUDGE"
log "  RUN_SCORE=$RUN_SCORE"
log "  HARD_K50_DATASET=$HARD_K50_DATASET"
log "  HARD_K100_DATASET=$HARD_K100_DATASET"
log "  SOFT_K50_DATASET=$SOFT_K50_DATASET"
log "  SOFT_K100_DATASET=$SOFT_K100_DATASET"

if [[ "$RUN_TRAIN" == "1" ]]; then
  "$SCRIPT_DIR/train_hard_k_50.sh"
  "$SCRIPT_DIR/train_hard_k_100.sh"
  "$SCRIPT_DIR/train_soft_k_50.sh"
  "$SCRIPT_DIR/train_soft_k_100.sh"
else
  log "Skipping training because RUN_TRAIN=$RUN_TRAIN."
fi

if [[ "$RUN_GENERATE" == "1" ]]; then
  "$SCRIPT_DIR/generate_hard.sh"
  "$SCRIPT_DIR/generate_soft.sh"
else
  log "Skipping response generation because RUN_GENERATE=$RUN_GENERATE."
fi

if [[ "$RUN_JUDGE" == "1" ]]; then
  "$SCRIPT_DIR/judge_hard.sh"
  "$SCRIPT_DIR/judge_soft.sh"
else
  log "Skipping judging because RUN_JUDGE=$RUN_JUDGE."
fi

if [[ "$RUN_SCORE" == "1" ]]; then
  "$SCRIPT_DIR/score_hard.sh"
  "$SCRIPT_DIR/score_soft.sh"
else
  log "Skipping scoring because RUN_SCORE=$RUN_SCORE."
fi

log "Finished PRISM k-sensitivity run."

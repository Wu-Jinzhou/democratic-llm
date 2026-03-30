#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_SET="${MODEL_SET:-us_uk}"
INCLUDE_BASE="${INCLUDE_BASE:-1}"
EVAL_DIR="${EVAL_DIR:-artifacts/evaluations/wvs/us_uk/llama3.1-8b}"
TOPK="${TOPK:-20}"
export MODEL_SET
export INCLUDE_BASE
export EVAL_DIR

# shellcheck disable=SC1091
source "$SCRIPT_DIR/../global_wvs/_wvs_eval_8b_common.sh"

mkdir -p "$EVAL_DIR"

log "UK/US filtered WVS evaluation configuration:"
log "  WVS_CSV=$WVS_CSV"
log "  QUESTIONS_JSON=$QUESTIONS_JSON"
log "  MODEL_SET=$MODEL_SET"
log "  INCLUDE_BASE=$INCLUDE_BASE"
log "  BATCH_SIZE=$BATCH_SIZE"
log "  MIN_RESPONDENTS=$MIN_RESPONDENTS"
log "  TOPK=$TOPK"
log "  EVAL_DIR=$EVAL_DIR"

CMD=(
  "$PYTHON" -u scripts/analysis/evaluate_wvs.py
  --wvs-csv "$WVS_CSV"
  --questions-json "$QUESTIONS_JSON"
  --models "${MODEL_LIST[@]}"
  --output-dir "$EVAL_DIR"
  --batch-size "$BATCH_SIZE"
  --min-respondents "$MIN_RESPONDENTS"
  --system-prompt "$SYSTEM_PROMPT"
  --topk "$TOPK"
)
if [[ ${#HF_ARGS[@]} -gt 0 ]]; then
  CMD+=("${HF_ARGS[@]}")
fi

log "Running filtered WVS evaluation for ${#MODEL_LIST[@]} models."
"${CMD[@]}"

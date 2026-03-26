#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_wvs_eval_8b_common.sh"

mkdir -p "$EVAL_DIR"

log "WVS evaluation configuration:"
log "  WVS_CSV=$WVS_CSV"
log "  QUESTIONS_JSON=$QUESTIONS_JSON"
log "  PANEL_CONFIG_OUTPUT=$PANEL_CONFIG_OUTPUT"
log "  PANEL_SIZE=$PANEL_SIZE"
log "  TOLERANCE=$TOLERANCE"
log "  MODEL_SET=$MODEL_SET"
log "  INCLUDE_BASE=$INCLUDE_BASE"
log "  BATCH_SIZE=$BATCH_SIZE"
log "  MIN_RESPONDENTS=$MIN_RESPONDENTS"
log "  EVAL_DIR=$EVAL_DIR"

log "Building WVS subjective-question file and global panel config."
"$PYTHON" -u scripts/build_wvs_assets.py \
  --wvs-csv "$WVS_CSV" \
  --questions-output "$QUESTIONS_JSON" \
  --panel-config-output "$PANEL_CONFIG_OUTPUT" \
  --panel-size "$PANEL_SIZE" \
  --tolerance "$TOLERANCE"

CMD=(
  "$PYTHON" -u scripts/analysis/evaluate_wvs.py
  --wvs-csv "$WVS_CSV"
  --questions-json "$QUESTIONS_JSON"
  --models "${MODEL_LIST[@]}"
  --output-dir "$EVAL_DIR"
  --batch-size "$BATCH_SIZE"
  --min-respondents "$MIN_RESPONDENTS"
  --system-prompt "$SYSTEM_PROMPT"
)
if [[ ${#HF_ARGS[@]} -gt 0 ]]; then
  CMD+=("${HF_ARGS[@]}")
fi

log "Running WVS evaluation for ${#MODEL_LIST[@]} models."
"${CMD[@]}"

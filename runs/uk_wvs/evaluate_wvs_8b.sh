#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_SET="${MODEL_SET:-us_uk}"
EVAL_DIR="${EVAL_DIR:-artifacts/evaluations/global_opinions/us_uk/llama3.1-8b}"
GLOBAL_OPINIONS_CSV="${GLOBAL_OPINIONS_CSV:-llm_global_opinions/data/global_opinions.csv}"
export MODEL_SET
export EVAL_DIR

# shellcheck disable=SC1091
source "$SCRIPT_DIR/../global_wvs/_wvs_eval_8b_common.sh"

mkdir -p "$EVAL_DIR"

log "UK/US GlobalOpinions evaluation configuration:"
log "  GLOBAL_OPINIONS_CSV=$GLOBAL_OPINIONS_CSV"
log "  MODEL_SET=$MODEL_SET"
log "  INCLUDE_BASE=$INCLUDE_BASE"
log "  BATCH_SIZE=$BATCH_SIZE"
log "  EVAL_DIR=$EVAL_DIR"

CMD=(
  "$PYTHON" -u scripts/analysis/evaluate_global_opinions.py
  --global-opinions-csv "$GLOBAL_OPINIONS_CSV"
  --models "${MODEL_LIST[@]}"
  --output-dir "$EVAL_DIR"
  --batch-size "$BATCH_SIZE"
  --system-prompt "$SYSTEM_PROMPT"
)
if [[ ${#HF_ARGS[@]} -gt 0 ]]; then
  CMD+=("${HF_ARGS[@]}")
fi

log "Running GlobalOpinions evaluation for ${#MODEL_LIST[@]} models."
"${CMD[@]}"

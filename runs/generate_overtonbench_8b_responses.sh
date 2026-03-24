#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_overtonbench_8b_common.sh"

log "Generating OvertonBench responses for ${#MODEL_LIST[@]} models."
log "  OVERTON_SOURCE=$OVERTON_SOURCE"
log "  RESPONSES_DIR=$RESPONSES_DIR"
log "  GEN_BATCH_SIZE=$GEN_BATCH_SIZE"
log "  MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
log "  SYSTEM_PROMPT=${SYSTEM_PROMPT:-<empty>}"
log "  MODELS=${MODEL_LIST[*]}"

cmd=(
  "$PYTHON" scripts/analysis/overtonbench_eval.py generate
  --questions-csv "$OVERTON_QUESTIONS_CSV"
  --source "$OVERTON_SOURCE"
  --models "${MODEL_LIST[@]}"
  --responses-dir "$RESPONSES_DIR"
  --batch-size "$GEN_BATCH_SIZE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --system-prompt "$SYSTEM_PROMPT"
)
if [[ ${#HF_ARGS[@]} -gt 0 ]]; then
  cmd+=("${HF_ARGS[@]}")
fi
"${cmd[@]}"

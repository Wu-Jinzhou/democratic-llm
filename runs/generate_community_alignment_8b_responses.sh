#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_community_alignment_eval_common.sh"

log "Generating Community Alignment evaluation responses."
log "  QUESTIONS_DIR=$QUESTIONS_DIR"
log "  RESPONSES_DIR=$RESPONSES_DIR"
log "  LISTWISE_PATH=$LISTWISE_PATH"
log "  GEN_BATCH_SIZE=$GEN_BATCH_SIZE"
log "  MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
log "  SYSTEM_PROMPT=${SYSTEM_PROMPT:-<empty>}"
log "  MODELS=${MODEL_LIST[*]}"

CMD=(
  "$PYTHON" -u scripts/evaluate_constitution.py
  --questions-dir "$QUESTIONS_DIR"
  --mode listwise
  --models "${MODEL_LIST[@]}"
  --responses-dir "$RESPONSES_DIR"
  --output "$LISTWISE_PATH"
  --preferences-output "$PREFERENCES_PATH"
  --batch-size "$GEN_BATCH_SIZE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --system-prompt "$SYSTEM_PROMPT"
  --skip-judging
)
CMD+=("${QUESTION_ARGS[@]}")
if [[ ${#HF_ARGS[@]} -gt 0 ]]; then
  CMD+=("${HF_ARGS[@]}")
fi
"${CMD[@]}"

log "Finished generating responses in $RESPONSES_DIR"

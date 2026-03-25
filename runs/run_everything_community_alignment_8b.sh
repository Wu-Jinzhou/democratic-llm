#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_community_alignment_common.sh"

RUN_PREP="${RUN_PREP:-0}"
RUN_GENERATION="${RUN_GENERATION:-1}"
WORLD_SIZE_OVERRIDE="${WORLD_SIZE_OVERRIDE:-${WORLD_SIZE:-1}}"

compute_matched_steps() {
  local dataset_path="$1"
  "$PYTHON" - <<'PY' "$dataset_path" "$PER_DEVICE_BATCH_SIZE" "$GRADIENT_ACCUMULATION_STEPS" "$NUM_TRAIN_EPOCHS" "$WORLD_SIZE_OVERRIDE"
import math
import sys
from pathlib import Path

dataset_path = Path(sys.argv[1])
per_device_batch_size = int(sys.argv[2])
grad_accum = int(sys.argv[3])
num_train_epochs = float(sys.argv[4])
world_size = int(sys.argv[5])

rows = 0
with dataset_path.open("r", encoding="utf-8") as f:
    for _ in f:
        rows += 1

micro_batches_per_epoch = math.ceil(rows / max(per_device_batch_size * world_size, 1))
update_steps = math.ceil(micro_batches_per_epoch * num_train_epochs / max(grad_accum, 1))
print(update_steps)
PY
}

log "Community Alignment end-to-end run configuration:"
log "  MODEL_ID=$MODEL_ID"
log "  PER_DEVICE_BATCH_SIZE=$PER_DEVICE_BATCH_SIZE"
log "  GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"
log "  NUM_TRAIN_EPOCHS=$NUM_TRAIN_EPOCHS"
log "  COMMUNITY_ALIGNMENT_NUM_TRAIN_STEPS=${COMMUNITY_ALIGNMENT_NUM_TRAIN_STEPS:-}"
log "  WORLD_SIZE_OVERRIDE=$WORLD_SIZE_OVERRIDE"
log "  RUN_PREP=$RUN_PREP"
log "  RUN_GENERATION=$RUN_GENERATION"
log "  QUESTIONS_PER_CLAUSE=${QUESTIONS_PER_CLAUSE:-40}"
log "  NUM_JUDGES=${NUM_JUDGES:-5}"
log "  JUDGE_MODEL=${JUDGE_MODEL:-gpt-5.2}"
log "  JUDGE_WORKERS=${JUDGE_WORKERS:-64}"
log "  GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-128}"

if [[ "$RUN_PREP" == "1" ]]; then
  "$SCRIPT_DIR/prepare_community_alignment.sh"
else
  log "Skipping dataset preparation because RUN_PREP=$RUN_PREP."
fi

"$SCRIPT_DIR/train_community_alignment_8b_full_us.sh"

if [[ -z "${COMMUNITY_ALIGNMENT_NUM_TRAIN_STEPS:-}" ]]; then
  COMMUNITY_ALIGNMENT_NUM_TRAIN_STEPS="$(compute_matched_steps "artifacts/data/community_alignment/full_us.jsonl")"
  export COMMUNITY_ALIGNMENT_NUM_TRAIN_STEPS
  log "Computed matched COMMUNITY_ALIGNMENT_NUM_TRAIN_STEPS=$COMMUNITY_ALIGNMENT_NUM_TRAIN_STEPS from full_us dataset size."
else
  log "Using provided COMMUNITY_ALIGNMENT_NUM_TRAIN_STEPS=$COMMUNITY_ALIGNMENT_NUM_TRAIN_STEPS."
fi

"$SCRIPT_DIR/train_community_alignment_8b_full_global.sh"
"$SCRIPT_DIR/train_community_alignment_8b_soft_us.sh"
"$SCRIPT_DIR/train_community_alignment_8b_hard_us.sh"

if [[ "$RUN_GENERATION" == "1" ]]; then
  "$SCRIPT_DIR/generate_community_alignment_8b_responses.sh"
else
  log "Skipping response generation because RUN_GENERATION=$RUN_GENERATION."
fi

log "Community Alignment train + generation run completed."

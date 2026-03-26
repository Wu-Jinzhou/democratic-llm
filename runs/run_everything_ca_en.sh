#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_community_alignment_common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_community_alignment_ca_en_eval_env.sh"

RUN_PREP="${RUN_PREP:-0}"
RUN_GENERATION="${RUN_GENERATION:-1}"
WORLD_SIZE_OVERRIDE="${WORLD_SIZE_OVERRIDE:-${WORLD_SIZE:-1}}"
OVERWRITE_RESPONSES="${OVERWRITE_RESPONSES:-0}"
export OVERWRITE_RESPONSES
RESPONSES_DIR="${RESPONSES_DIR:-$EVAL_DIR/responses}"

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

safe_model_id() {
  local model_id="$1"
  printf '%s' "${model_id//\//__}" | tr ':' '_'
}

log "Community Alignment EN run configuration:"
log "  MODEL_ID=$MODEL_ID"
log "  RUN_PREP=$RUN_PREP"
log "  RUN_GENERATION=$RUN_GENERATION"
log "  PER_DEVICE_BATCH_SIZE=$PER_DEVICE_BATCH_SIZE"
log "  GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"
log "  NUM_TRAIN_EPOCHS=$NUM_TRAIN_EPOCHS"
log "  COMMUNITY_ALIGNMENT_NUM_TRAIN_STEPS=${COMMUNITY_ALIGNMENT_NUM_TRAIN_STEPS:-}"
log "  OVERWRITE_RESPONSES=$OVERWRITE_RESPONSES"
log "  SOFT_MODEL=$SOFT_MODEL"
log "  HARD_MODEL=$HARD_MODEL"
log "  FULL_US_MODEL=$FULL_US_MODEL"
log "  FULL_EN_GLOBAL_MODEL=$FULL_EN_GLOBAL_MODEL"
log "  EVAL_DIR=$EVAL_DIR"
log "  RESPONSES_DIR=$RESPONSES_DIR"

if [[ "$RUN_PREP" == "1" ]]; then
  "$SCRIPT_DIR/prepare_community_alignment.sh"
else
  log "Skipping dataset preparation because RUN_PREP=$RUN_PREP."
fi

if [[ -z "${COMMUNITY_ALIGNMENT_NUM_TRAIN_STEPS:-}" ]]; then
  COMMUNITY_ALIGNMENT_NUM_TRAIN_STEPS="$(compute_matched_steps "artifacts/data/community_alignment/full_us.jsonl")"
  export COMMUNITY_ALIGNMENT_NUM_TRAIN_STEPS
  log "Computed matched COMMUNITY_ALIGNMENT_NUM_TRAIN_STEPS=$COMMUNITY_ALIGNMENT_NUM_TRAIN_STEPS from English-subset full_us dataset size."
else
  log "Using provided COMMUNITY_ALIGNMENT_NUM_TRAIN_STEPS=$COMMUNITY_ALIGNMENT_NUM_TRAIN_STEPS."
fi

"$SCRIPT_DIR/train_community_alignment_8b_full_en_global.sh"
"$SCRIPT_DIR/train_community_alignment_8b_soft_en.sh"
"$SCRIPT_DIR/train_community_alignment_8b_hard_en.sh"

if [[ "$RUN_GENERATION" == "1" ]]; then
  mkdir -p "$RESPONSES_DIR"
  for model_id in "$FULL_EN_GLOBAL_MODEL" "$SOFT_MODEL" "$HARD_MODEL"; do
    responses_path="$RESPONSES_DIR/$(safe_model_id "$model_id").jsonl"
    if [[ -f "$responses_path" ]]; then
      log "Removing stale cached responses for retrained model: $responses_path"
      rm -f "$responses_path"
    fi
  done
  "$SCRIPT_DIR/generate_ca_en_responses.sh"
else
  log "Skipping response generation because RUN_GENERATION=$RUN_GENERATION."
fi

log "Community Alignment EN train + generation run completed."

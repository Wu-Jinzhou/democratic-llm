#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_community_alignment_common.sh"

DATASET="${DATASET:-artifacts/data/community_alignment/hard_panel_en.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/community-alignment/llama3.1-8b-hard-en}"

require_num_train_steps

log "Training Community Alignment hard-panel English-pool model."
log "  MODEL_ID=$MODEL_ID"
log "  DATASET=$DATASET"
log "  OUTPUT_DIR=$OUTPUT_DIR"
log "  NUM_TRAIN_STEPS=$NUM_TRAIN_STEPS"
log "  PER_DEVICE_BATCH_SIZE=$PER_DEVICE_BATCH_SIZE"
log "  GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"
log "  ATTN_IMPLEMENTATION=$ATTN_IMPLEMENTATION"

CMD=()
build_common_train_cmd CMD
CMD+=(
  --dataset "$DATASET"
  --output-dir "$OUTPUT_DIR"
  --num-train-steps "$NUM_TRAIN_STEPS"
)
"${CMD[@]}"

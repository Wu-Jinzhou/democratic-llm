#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_community_alignment_common.sh"

DATASET="${DATASET:-artifacts/data/community_alignment/full_us.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/community-alignment/llama3.1-8b-full-us}"

log "Training Community Alignment English-subset US-full model."
log "  MODEL_ID=$MODEL_ID"
log "  DATASET=$DATASET"
log "  OUTPUT_DIR=$OUTPUT_DIR"
log "  PER_DEVICE_BATCH_SIZE=$PER_DEVICE_BATCH_SIZE"
log "  GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"
log "  ATTN_IMPLEMENTATION=$ATTN_IMPLEMENTATION"
log "  REPORT_TO=$REPORT_TO"
log "  WANDB_PROJECT=$WANDB_PROJECT"

CMD=()
build_common_train_cmd CMD
CMD+=(
  --dataset "$DATASET"
  --output-dir "$OUTPUT_DIR"
  --num-train-epochs "$NUM_TRAIN_EPOCHS"
)
"${CMD[@]}"

echo "US-full training completed."
echo "Use its observed 2-epoch optimizer step count as COMMUNITY_ALIGNMENT_NUM_TRAIN_STEPS for the global/soft/hard runs."

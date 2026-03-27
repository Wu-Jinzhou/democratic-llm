#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_prism_uk_wvs_common.sh"

DATASET="${DATASET:-artifacts/data/uk_rep.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/llama3.1-8b-uk-rep}"

log "Training PRISM UK representative model."
log "  MODEL_ID=$MODEL_ID"
log "  DATASET=$DATASET"
log "  OUTPUT_DIR=$OUTPUT_DIR"
log "  NUM_TRAIN_STEPS=$NUM_TRAIN_STEPS"
log "  PER_DEVICE_BATCH_SIZE=$PER_DEVICE_BATCH_SIZE"
log "  GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"
log "  ATTN_IMPLEMENTATION=$ATTN_IMPLEMENTATION"
log "  REPORT_TO=$REPORT_TO"
log "  WANDB_PROJECT=$WANDB_PROJECT"

TRAIN_CMD=()
build_common_train_cmd TRAIN_CMD
TRAIN_CMD+=(--dataset "$DATASET" --output-dir "$OUTPUT_DIR")
"${TRAIN_CMD[@]}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B}"
SHORT="${SHORT:-prism-k-sensitivity-llama3.1-8b}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-3538}"
COMMUNITY_ALIGNMENT_NUM_TRAIN_STEPS="${COMMUNITY_ALIGNMENT_NUM_TRAIN_STEPS:-$NUM_TRAIN_STEPS}"
WANDB_GROUP="${WANDB_GROUP:-prism-k-sensitivity-8b}"

# shellcheck disable=SC1091
source "$RUNS_ROOT/community_alignment/_community_alignment_common.sh"

HARD_BASE_MODEL="${HARD_BASE_MODEL:-checkpoints/llama3.1-8b-hard-panel}"
SOFT_BASE_MODEL="${SOFT_BASE_MODEL:-checkpoints/llama3.1-8b-soft-panel}"

HARD_K50_DATASET="${HARD_K50_DATASET:-artifacts/data/hard_panel_control_k_50.jsonl}"
HARD_K100_DATASET="${HARD_K100_DATASET:-artifacts/data/hard_panel_control_k_100.jsonl}"
SOFT_K50_DATASET="${SOFT_K50_DATASET:-artifacts/data/soft_panel_control_k_50.jsonl}"
SOFT_K100_DATASET="${SOFT_K100_DATASET:-artifacts/data/soft_panel_control_k_100.jsonl}"

HARD_K50_MODEL="${HARD_K50_MODEL:-checkpoints/llama3.1-8b-hard-panel-control-k50}"
HARD_K100_MODEL="${HARD_K100_MODEL:-checkpoints/llama3.1-8b-hard-panel-control-k100}"
SOFT_K50_MODEL="${SOFT_K50_MODEL:-checkpoints/llama3.1-8b-soft-panel-control-k50}"
SOFT_K100_MODEL="${SOFT_K100_MODEL:-checkpoints/llama3.1-8b-soft-panel-control-k100}"

require_dataset() {
  local dataset_path="$1"
  if [[ ! -f "$dataset_path" ]]; then
    echo "Missing dataset: $dataset_path" >&2
    exit 1
  fi
}

run_train() {
  local dataset_path="$1"
  local output_dir="$2"
  local label="$3"

  require_num_train_steps
  require_dataset "$dataset_path"

  log "$label"
  log "  MODEL_ID=$MODEL_ID"
  log "  DATASET=$dataset_path"
  log "  OUTPUT_DIR=$output_dir"
  log "  NUM_TRAIN_STEPS=$NUM_TRAIN_STEPS"
  log "  PER_DEVICE_BATCH_SIZE=$PER_DEVICE_BATCH_SIZE"
  log "  GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"
  log "  ATTN_IMPLEMENTATION=$ATTN_IMPLEMENTATION"

  CMD=()
  build_common_train_cmd CMD
  CMD+=(
    --dataset "$dataset_path"
    --output-dir "$output_dir"
    --num-train-steps "$NUM_TRAIN_STEPS"
  )
  "${CMD[@]}"
}

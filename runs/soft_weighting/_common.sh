#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B}"
SHORT="${SHORT:-prism-soft-weighting-llama3.1-8b}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-3538}"
WANDB_GROUP="${WANDB_GROUP:-prism-soft-weighting-8b}"

# shellcheck disable=SC1091
source "$RUNS_ROOT/global_wvs/_prism_global_wvs_common.sh"

PANEL_CONFIG="${PANEL_CONFIG:-configs/panel_config.yaml}"
SOFT_LINEAR_DATASET="${SOFT_LINEAR_DATASET:-artifacts/data/soft_panel.jsonl}"
SOFT_SQRT_DATASET="${SOFT_SQRT_DATASET:-artifacts/data/soft_panel_sqrt.jsonl}"
SOFT_SQUARE_DATASET="${SOFT_SQUARE_DATASET:-artifacts/data/soft_panel_square.jsonl}"
SOFT_CLIPPED_DATASET="${SOFT_CLIPPED_DATASET:-artifacts/data/soft_panel_clipped.jsonl}"

SOFT_LINEAR_MODEL="${SOFT_LINEAR_MODEL:-checkpoints/llama3.1-8b-soft-panel}"
SOFT_SQRT_MODEL="${SOFT_SQRT_MODEL:-checkpoints/llama3.1-8b-soft-panel-sqrt}"
SOFT_SQUARE_MODEL="${SOFT_SQUARE_MODEL:-checkpoints/llama3.1-8b-soft-panel-square}"
SOFT_CLIPPED_MODEL="${SOFT_CLIPPED_MODEL:-checkpoints/llama3.1-8b-soft-panel-clipped}"
FULL_MODEL="${FULL_MODEL:-checkpoints/llama3.1-8b-full-prism}"
HARD_MODEL="${HARD_MODEL:-checkpoints/llama3.1-8b-hard-panel}"

CLIP_MIN="${CLIP_MIN:-}"
CLIP_MAX="${CLIP_MAX:-}"
CLIP_LOWER_QUANTILE="${CLIP_LOWER_QUANTILE:-0.05}"
CLIP_UPPER_QUANTILE="${CLIP_UPPER_QUANTILE:-0.95}"
PANEL_SEED="${PANEL_SEED:-42}"
NUM_PANEL_SAMPLES="${NUM_PANEL_SAMPLES:-2000}"
NUM_WORKERS_PREP="${NUM_WORKERS_PREP:-12}"
DATASET_FORMAT="${DATASET_FORMAT:-chat}"
USE_CONVERSATIONS="${USE_CONVERSATIONS:-1}"
DELTA="${DELTA:-0.0}"
RATER_NORMALIZATION="${RATER_NORMALIZATION:-panel}"

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
  )
  "${CMD[@]}"
}

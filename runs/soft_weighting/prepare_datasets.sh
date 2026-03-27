#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

build_prepare_cmd() {
  local out_name="$1"
  local output_path="$2"
  local transform="$3"

  local cmd=(
    "$PYTHON" scripts/prepare_data.py
    --survey prism-alignment/survey.jsonl
    --utterances prism-alignment/utterances.jsonl
    --conversations prism-alignment/conversations.jsonl
    --mode soft
    --panel-config "$PANEL_CONFIG"
    --panel-algorithm leximin
    --panel-seed "$PANEL_SEED"
    --num-panel-samples "$NUM_PANEL_SAMPLES"
    --num-workers "$NUM_WORKERS_PREP"
    --dataset-format "$DATASET_FORMAT"
    --delta "$DELTA"
    --rater-normalization "$RATER_NORMALIZATION"
    --soft-weight-transform "$transform"
    --soft-weight-clip-lower-quantile "$CLIP_LOWER_QUANTILE"
    --soft-weight-clip-upper-quantile "$CLIP_UPPER_QUANTILE"
    --output "$output_path"
  )
  if [[ "$USE_CONVERSATIONS" == "1" ]]; then
    cmd+=(--use-conversations)
  else
    cmd+=(--no-use-conversations)
  fi
  if [[ -n "$CLIP_MIN" ]]; then
    cmd+=(--soft-weight-clip-min "$CLIP_MIN")
  fi
  if [[ -n "$CLIP_MAX" ]]; then
    cmd+=(--soft-weight-clip-max "$CLIP_MAX")
  fi
  eval "$out_name=()"
  local arg quoted_arg
  for arg in "${cmd[@]}"; do
    printf -v quoted_arg '%q' "$arg"
    eval "$out_name+=( $quoted_arg )"
  done
}

log "Preparing PRISM soft-panel weight-transform datasets."
log "  PANEL_CONFIG=$PANEL_CONFIG"
log "  PANEL_SEED=$PANEL_SEED"
log "  NUM_PANEL_SAMPLES=$NUM_PANEL_SAMPLES"
log "  NUM_WORKERS_PREP=$NUM_WORKERS_PREP"
log "  SOFT_SQRT_DATASET=$SOFT_SQRT_DATASET"
log "  SOFT_SQUARE_DATASET=$SOFT_SQUARE_DATASET"
log "  SOFT_CLIPPED_DATASET=$SOFT_CLIPPED_DATASET"
log "  CLIP_MIN=${CLIP_MIN:-<quantile-derived>}"
log "  CLIP_MAX=${CLIP_MAX:-<quantile-derived>}"
log "  CLIP_LOWER_QUANTILE=$CLIP_LOWER_QUANTILE"
log "  CLIP_UPPER_QUANTILE=$CLIP_UPPER_QUANTILE"

CMD=()
build_prepare_cmd CMD "$SOFT_SQRT_DATASET" "sqrt"
"${CMD[@]}"

build_prepare_cmd CMD "$SOFT_SQUARE_DATASET" "square"
"${CMD[@]}"

build_prepare_cmd CMD "$SOFT_CLIPPED_DATASET" "clip"
"${CMD[@]}"

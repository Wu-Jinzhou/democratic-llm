#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PYTHON="${PYTHON:-python}"

INPUT_CSV="${INPUT_CSV:-community-alignment-dataset/community_alignment.csv}"
NORMALIZED_DIR="${NORMALIZED_DIR:-artifacts/community_alignment/normalized}"
DATA_DIR="${DATA_DIR:-artifacts/data/community_alignment}"

PANEL_CONFIG="${PANEL_CONFIG:-configs/panel_config_community_alignment_us.yaml}"
PANEL_ALGORITHM="${PANEL_ALGORITHM:-leximin}"
PANEL_SEED="${PANEL_SEED:-42}"
NUM_PANEL_SAMPLES="${NUM_PANEL_SAMPLES:-2000}"
NUM_WORKERS="${NUM_WORKERS:-32}"

DATASET_FORMAT="${DATASET_FORMAT:-chat}"
USE_CONVERSATIONS="${USE_CONVERSATIONS:-1}"
DELTA="${DELTA:-0.0}"
RATER_NORMALIZATION="${RATER_NORMALIZATION:-panel}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"

mkdir -p "$NORMALIZED_DIR" "$DATA_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_step() {
  local label="$1"
  shift
  log "$label"
  "$@"
}

CONV_ARGS=()
if [[ "$USE_CONVERSATIONS" == "1" ]]; then
  CONV_ARGS+=(--use-conversations)
else
  CONV_ARGS+=(--no-use-conversations)
fi

SYSTEM_PROMPT_ARGS=()
if [[ -n "$SYSTEM_PROMPT" ]]; then
  SYSTEM_PROMPT_ARGS+=(--system-prompt "$SYSTEM_PROMPT")
fi

log "Community Alignment prep configuration:"
log "  INPUT_CSV=$INPUT_CSV"
log "  NORMALIZED_DIR=$NORMALIZED_DIR"
log "  DATA_DIR=$DATA_DIR"
log "  PANEL_CONFIG=$PANEL_CONFIG"
log "  PANEL_ALGORITHM=$PANEL_ALGORITHM"
log "  PANEL_SEED=$PANEL_SEED"
log "  NUM_PANEL_SAMPLES=$NUM_PANEL_SAMPLES"
log "  NUM_WORKERS=$NUM_WORKERS"
log "  DATASET_FORMAT=$DATASET_FORMAT"
log "  USE_CONVERSATIONS=$USE_CONVERSATIONS"
log "  DELTA=$DELTA"
log "  RATER_NORMALIZATION=$RATER_NORMALIZATION"

if [[ "$PANEL_ALGORITHM" == "leximin" ]]; then
  run_step "Checking Gurobi availability for LEXIMIN." "$PYTHON" -u - <<'PY'
try:
    import gurobipy  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PANEL_ALGORITHM=leximin requires gurobipy in the active environment. "
        "Install gurobipy and ensure a valid Gurobi license is available, or override "
        "PANEL_ALGORITHM to legacy if you want the approximate path."
    ) from exc
PY
fi

run_step "Converting raw Community Alignment CSV into normalized JSONL files." \
  "$PYTHON" -u scripts/convert_community_alignment.py \
  --input "$INPUT_CSV" \
  --output-dir "$NORMALIZED_DIR"

run_step "Preparing global full dataset." \
  "$PYTHON" -u scripts/prepare_data.py \
  --survey "$NORMALIZED_DIR/survey.jsonl" \
  --utterances "$NORMALIZED_DIR/utterances.jsonl" \
  --conversations "$NORMALIZED_DIR/conversations.jsonl" \
  --mode full \
  --dataset-format "$DATASET_FORMAT" \
  "${CONV_ARGS[@]}" \
  --delta "$DELTA" \
  --rater-normalization "$RATER_NORMALIZATION" \
  "${SYSTEM_PROMPT_ARGS[@]}" \
  --output "$DATA_DIR/full_global.jsonl"

run_step "Preparing US full dataset." \
  "$PYTHON" -u scripts/prepare_data.py \
  --survey "$NORMALIZED_DIR/survey.jsonl" \
  --utterances "$NORMALIZED_DIR/utterances.jsonl" \
  --conversations "$NORMALIZED_DIR/conversations.jsonl" \
  --mode us_rep \
  --dataset-format "$DATASET_FORMAT" \
  "${CONV_ARGS[@]}" \
  --delta "$DELTA" \
  --rater-normalization "$RATER_NORMALIZATION" \
  "${SYSTEM_PROMPT_ARGS[@]}" \
  --output "$DATA_DIR/full_us.jsonl"

run_step "Preparing US soft-panel dataset." \
  "$PYTHON" -u scripts/prepare_data.py \
  --survey "$NORMALIZED_DIR/survey.jsonl" \
  --utterances "$NORMALIZED_DIR/utterances.jsonl" \
  --conversations "$NORMALIZED_DIR/conversations.jsonl" \
  --mode soft \
  --panel-config "$PANEL_CONFIG" \
  --panel-algorithm "$PANEL_ALGORITHM" \
  --panel-seed "$PANEL_SEED" \
  --num-panel-samples "$NUM_PANEL_SAMPLES" \
  --num-workers "$NUM_WORKERS" \
  --dataset-format "$DATASET_FORMAT" \
  "${CONV_ARGS[@]}" \
  --delta "$DELTA" \
  --rater-normalization "$RATER_NORMALIZATION" \
  "${SYSTEM_PROMPT_ARGS[@]}" \
  --output "$DATA_DIR/soft_panel_us.jsonl"

run_step "Preparing US hard-panel dataset." \
  "$PYTHON" -u scripts/prepare_data.py \
  --survey "$NORMALIZED_DIR/survey.jsonl" \
  --utterances "$NORMALIZED_DIR/utterances.jsonl" \
  --conversations "$NORMALIZED_DIR/conversations.jsonl" \
  --mode hard \
  --panel-config "$PANEL_CONFIG" \
  --panel-algorithm "$PANEL_ALGORITHM" \
  --panel-seed "$PANEL_SEED" \
  --num-workers "$NUM_WORKERS" \
  --dataset-format "$DATASET_FORMAT" \
  "${CONV_ARGS[@]}" \
  --delta "$DELTA" \
  --rater-normalization "$RATER_NORMALIZATION" \
  "${SYSTEM_PROMPT_ARGS[@]}" \
  --output "$DATA_DIR/hard_panel_us.jsonl"

log "Prepared Community Alignment datasets in $DATA_DIR"

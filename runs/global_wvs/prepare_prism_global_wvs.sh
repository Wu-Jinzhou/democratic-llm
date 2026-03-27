#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PYTHON="${PYTHON:-python}"

PANEL_CONFIG="${PANEL_CONFIG:-configs/panel_config_global.yaml}"

SURVEY_PATH="${SURVEY_PATH:-prism-alignment/survey.jsonl}"
UTTERANCES_PATH="${UTTERANCES_PATH:-prism-alignment/utterances.jsonl}"
CONVERSATIONS_PATH="${CONVERSATIONS_PATH:-prism-alignment/conversations.jsonl}"

DATA_DIR="${DATA_DIR:-artifacts/data}"
SOFT_OUTPUT="${SOFT_OUTPUT:-$DATA_DIR/global_wvs_soft_panel.jsonl}"
HARD_OUTPUT="${HARD_OUTPUT:-$DATA_DIR/global_wvs_hard_panel.jsonl}"

PANEL_ALGORITHM="${PANEL_ALGORITHM:-leximin}"
PANEL_SEED="${PANEL_SEED:-42}"
NUM_PANEL_SAMPLES="${NUM_PANEL_SAMPLES:-2000}"
NUM_WORKERS="${NUM_WORKERS:-12}"

DATASET_FORMAT="${DATASET_FORMAT:-chat}"
USE_CONVERSATIONS="${USE_CONVERSATIONS:-1}"
DELTA="${DELTA:-0.0}"
RATER_NORMALIZATION="${RATER_NORMALIZATION:-panel}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_step() {
  local label="$1"
  shift
  log "$label"
  "$@"
}

mkdir -p "$DATA_DIR"

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

log "PRISM global WVS prep configuration:"
log "  PANEL_CONFIG=$PANEL_CONFIG"
log "  SURVEY_PATH=$SURVEY_PATH"
log "  UTTERANCES_PATH=$UTTERANCES_PATH"
log "  CONVERSATIONS_PATH=$CONVERSATIONS_PATH"
log "  SOFT_OUTPUT=$SOFT_OUTPUT"
log "  HARD_OUTPUT=$HARD_OUTPUT"
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

SOFT_CMD=(
  "$PYTHON" -u scripts/prepare_data.py
  --survey "$SURVEY_PATH"
  --utterances "$UTTERANCES_PATH"
  --conversations "$CONVERSATIONS_PATH"
  --mode soft
  --panel-config "$PANEL_CONFIG"
  --panel-algorithm "$PANEL_ALGORITHM"
  --panel-seed "$PANEL_SEED"
  --num-panel-samples "$NUM_PANEL_SAMPLES"
  --num-workers "$NUM_WORKERS"
  --dataset-format "$DATASET_FORMAT"
  "${CONV_ARGS[@]}"
  --delta "$DELTA"
  --rater-normalization "$RATER_NORMALIZATION"
  --output "$SOFT_OUTPUT"
)
if [[ ${#SYSTEM_PROMPT_ARGS[@]} -gt 0 ]]; then
  SOFT_CMD+=("${SYSTEM_PROMPT_ARGS[@]}")
fi
run_step "Preparing PRISM global WVS soft-panel dataset." "${SOFT_CMD[@]}"

HARD_CMD=(
  "$PYTHON" -u scripts/prepare_data.py
  --survey "$SURVEY_PATH"
  --utterances "$UTTERANCES_PATH"
  --conversations "$CONVERSATIONS_PATH"
  --mode hard
  --panel-config "$PANEL_CONFIG"
  --panel-algorithm "$PANEL_ALGORITHM"
  --panel-seed "$PANEL_SEED"
  --num-workers "$NUM_WORKERS"
  --dataset-format "$DATASET_FORMAT"
  "${CONV_ARGS[@]}"
  --delta "$DELTA"
  --rater-normalization "$RATER_NORMALIZATION"
  --output "$HARD_OUTPUT"
)
if [[ ${#SYSTEM_PROMPT_ARGS[@]} -gt 0 ]]; then
  HARD_CMD+=("${SYSTEM_PROMPT_ARGS[@]}")
fi
run_step "Preparing PRISM global WVS hard-panel dataset." "${HARD_CMD[@]}"

log "Prepared PRISM global WVS datasets: $SOFT_OUTPUT and $HARD_OUTPUT"

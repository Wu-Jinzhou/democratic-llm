#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PYTHON="${PYTHON:-python}"

BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B}"
INCLUDE_BASE="${INCLUDE_BASE:-0}"
MODEL_SET="${MODEL_SET:-all}"

PRISM_FULL_MODEL="${PRISM_FULL_MODEL:-checkpoints/llama3.1-8b-full-prism}"
PRISM_SOFT_MODEL="${PRISM_SOFT_MODEL:-checkpoints/llama3.1-8b-soft-panel}"
PRISM_HARD_MODEL="${PRISM_HARD_MODEL:-checkpoints/llama3.1-8b-hard-panel}"
PRISM_US_REP_MODEL="${PRISM_US_REP_MODEL:-checkpoints/llama3.1-8b-us-rep}"
PRISM_UK_REP_MODEL="${PRISM_UK_REP_MODEL:-checkpoints/llama3.1-8b-uk-rep}"
PRISM_UK_SOFT_MODEL="${PRISM_UK_SOFT_MODEL:-checkpoints/llama3.1-8b-uk-soft-panel}"
PRISM_UK_HARD_MODEL="${PRISM_UK_HARD_MODEL:-checkpoints/llama3.1-8b-uk-hard-panel}"

PRISM_GLOBAL_WVS_SOFT_MODEL="${PRISM_GLOBAL_WVS_SOFT_MODEL:-checkpoints/llama3.1-8b-global-wvs-soft-panel}"
PRISM_GLOBAL_WVS_HARD_MODEL="${PRISM_GLOBAL_WVS_HARD_MODEL:-checkpoints/llama3.1-8b-global-wvs-hard-panel}"

WVS_CSV="${WVS_CSV:-wvs/WVS_Cross-National_Wave_7_csv_v6_0.csv}"
QUESTIONS_JSON="${QUESTIONS_JSON:-wvs/subjective_questions.json}"

BATCH_SIZE="${BATCH_SIZE:-64}"
MIN_RESPONDENTS="${MIN_RESPONDENTS:-100}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"

HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

EVAL_DIR="${EVAL_DIR:-artifacts/evaluations/wvs/llama3.1-8b}"

case "$MODEL_SET" in
  us)
    MODEL_LIST=(
      "$PRISM_FULL_MODEL"
      "$PRISM_SOFT_MODEL"
      "$PRISM_HARD_MODEL"
      "$PRISM_US_REP_MODEL"
    )
    ;;
  global)
    MODEL_LIST=(
      "$PRISM_FULL_MODEL"
      "$PRISM_GLOBAL_WVS_SOFT_MODEL"
      "$PRISM_GLOBAL_WVS_HARD_MODEL"
    )
    ;;
  uk)
    MODEL_LIST=(
      "$PRISM_FULL_MODEL"
      "$PRISM_UK_REP_MODEL"
      "$PRISM_UK_SOFT_MODEL"
      "$PRISM_UK_HARD_MODEL"
    )
    ;;
  us_uk)
    MODEL_LIST=(
      "$PRISM_FULL_MODEL"
      "$PRISM_SOFT_MODEL"
      "$PRISM_HARD_MODEL"
      "$PRISM_US_REP_MODEL"
      "$PRISM_UK_REP_MODEL"
      "$PRISM_UK_SOFT_MODEL"
      "$PRISM_UK_HARD_MODEL"
    )
    ;;
  all)
    MODEL_LIST=(
      "$PRISM_FULL_MODEL"
      "$PRISM_SOFT_MODEL"
      "$PRISM_HARD_MODEL"
      "$PRISM_US_REP_MODEL"
      "$PRISM_UK_REP_MODEL"
      "$PRISM_UK_SOFT_MODEL"
      "$PRISM_UK_HARD_MODEL"
      "$PRISM_GLOBAL_WVS_SOFT_MODEL"
      "$PRISM_GLOBAL_WVS_HARD_MODEL"
    )
    ;;
  *)
    echo "Invalid MODEL_SET=$MODEL_SET. Expected one of: us, uk, us_uk, global, all." >&2
    exit 1
    ;;
esac

if [[ "$INCLUDE_BASE" == "1" ]]; then
  MODEL_LIST=("$BASE_MODEL" "${MODEL_LIST[@]}")
fi

HF_ARGS=()
if [[ -n "$HF_TOKEN" ]]; then
  HF_ARGS+=(--hf-token "$HF_TOKEN")
fi

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

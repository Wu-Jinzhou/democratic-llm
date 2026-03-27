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
FULL_EN_GLOBAL_MODEL="${FULL_EN_GLOBAL_MODEL:-${FULL_GLOBAL_MODEL:-checkpoints/community-alignment/llama3.1-8b-full-en-global}}"
FULL_US_MODEL="${FULL_US_MODEL:-checkpoints/community-alignment/llama3.1-8b-full-us}"
SOFT_MODEL="${SOFT_MODEL:-${SOFT_EN_MODEL:-${SOFT_US_MODEL:-checkpoints/community-alignment/llama3.1-8b-soft-en}}}"
HARD_MODEL="${HARD_MODEL:-${HARD_EN_MODEL:-${HARD_US_MODEL:-checkpoints/community-alignment/llama3.1-8b-hard-en}}}"

QUESTIONS_DIR="${QUESTIONS_DIR:-artifacts/questions}"
QUESTIONS_PER_CLAUSE="${QUESTIONS_PER_CLAUSE:-40}"
MAX_QUESTIONS="${MAX_QUESTIONS:-}"

GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

NUM_JUDGES="${NUM_JUDGES:-5}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.2}"
JUDGE_WORKERS="${JUDGE_WORKERS:-64}"
JUDGE_RETRIES="${JUDGE_RETRIES:-10}"
RETRY_BACKOFF="${RETRY_BACKOFF:-2.0}"
JUDGE_MAX_OUTPUT_TOKENS="${JUDGE_MAX_OUTPUT_TOKENS:-400}"

SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
OVERWRITE_RESPONSES="${OVERWRITE_RESPONSES:-0}"

EVAL_DIR="${EVAL_DIR:-artifacts/evaluations/community-alignment/llama3.1-8b/no_system_prompt}"
RESPONSES_DIR="${RESPONSES_DIR:-$EVAL_DIR/responses}"
LISTWISE_PATH="${LISTWISE_PATH:-$EVAL_DIR/listwise.jsonl}"
PREFERENCES_PATH="${PREFERENCES_PATH:-$EVAL_DIR/preferences.jsonl}"

MODEL_LIST=(
  "$BASE_MODEL"
  "$FULL_EN_GLOBAL_MODEL"
  "$FULL_US_MODEL"
  "$SOFT_MODEL"
  "$HARD_MODEL"
)

mkdir -p "$EVAL_DIR" "$RESPONSES_DIR"

HF_ARGS=()
if [[ -n "$HF_TOKEN" ]]; then
  HF_ARGS+=(--hf-token "$HF_TOKEN")
fi

QUESTION_ARGS=()
if [[ -n "$MAX_QUESTIONS" ]]; then
  QUESTION_ARGS+=(--max-questions "$MAX_QUESTIONS")
else
  QUESTION_ARGS+=(--questions-per-clause "$QUESTIONS_PER_CLAUSE")
fi

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

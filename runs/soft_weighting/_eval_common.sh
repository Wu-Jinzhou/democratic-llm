#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PYTHON="${PYTHON:-python}"

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

EVAL_DIR="${EVAL_DIR:-artifacts/evaluations/prism/soft-weighting/llama3.1-8b/no_system_prompt}"
RESPONSES_DIR="${RESPONSES_DIR:-artifacts/evaluations/prism/llama3.1-8b/no_system_prompt/responses}"
LISTWISE_PATH="${LISTWISE_PATH:-$EVAL_DIR/listwise.jsonl}"
PREFERENCES_PATH="${PREFERENCES_PATH:-$EVAL_DIR/preferences.jsonl}"

MODEL_LIST=(
  "$SOFT_LINEAR_MODEL"
  "$SOFT_SQRT_MODEL"
  "$SOFT_SQUARE_MODEL"
  "$SOFT_CLIPPED_MODEL"
  "$FULL_MODEL"
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

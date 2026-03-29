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
FULL_US_MODEL="${FULL_US_MODEL:-checkpoints/community-alignment/llama3.1-8b-full-us}"
US_BALANCED_MODEL="${US_BALANCED_MODEL:-checkpoints/community-alignment/llama3.1-8b-us-balanced-subset}"
SOFT_MODEL="${SOFT_MODEL:-checkpoints/community-alignment/llama3.1-8b-soft-us-k350}"
HARD_MODEL="${HARD_MODEL:-checkpoints/community-alignment/llama3.1-8b-hard-us-k350}"

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

EVAL_DIR="${EVAL_DIR:-artifacts/evaluations/community-alignment-us-balanced/llama3.1-8b/no_system_prompt}"
RESPONSES_DIR="${RESPONSES_DIR:-$EVAL_DIR/responses}"
LISTWISE_PATH="${LISTWISE_PATH:-$EVAL_DIR/listwise.jsonl}"
PREFERENCES_PATH="${PREFERENCES_PATH:-$EVAL_DIR/preferences.jsonl}"

LEGACY_RESPONSES_DIR="${LEGACY_RESPONSES_DIR:-artifacts/evaluations/community-alignment/llama3.1-8b/no_system_prompt/responses}"
BASE_RESPONSE_FILE="${BASE_RESPONSE_FILE:-meta-llama__Llama-3.1-8B.jsonl}"
FULL_US_RESPONSE_FILE="${FULL_US_RESPONSE_FILE:-checkpoints__community-alignment__llama3.1-8b-full-us.jsonl}"

MODEL_LIST=(
  "$BASE_MODEL"
  "$FULL_US_MODEL"
  "$US_BALANCED_MODEL"
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

ensure_ca_us_balanced_cached_responses() {
  local target_dir="${RESPONSES_DIR:-$EVAL_DIR/responses}"
  mkdir -p "$target_dir"
  local src dst
  for file in "$BASE_RESPONSE_FILE" "$FULL_US_RESPONSE_FILE"; do
    src="$LEGACY_RESPONSES_DIR/$file"
    dst="$target_dir/$file"
    if [[ ! -f "$dst" && -f "$src" ]]; then
      cp "$src" "$dst"
    fi
  done
}

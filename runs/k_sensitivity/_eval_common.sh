#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Preserve any explicit caller overrides before sourcing the shared CA eval env,
# which otherwise populates Community Alignment defaults we do not want here.
USER_EVAL_DIR="${EVAL_DIR:-}"
USER_RESPONSES_DIR="${RESPONSES_DIR:-}"
USER_LISTWISE_PATH="${LISTWISE_PATH:-}"
USER_PREFERENCES_PATH="${PREFERENCES_PATH:-}"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"
# shellcheck disable=SC1091
source "$RUNS_ROOT/community_alignment/_community_alignment_eval_common.sh"

EVAL_ROOT="${EVAL_ROOT:-artifacts/evaluations/prism/k-sensitivity/llama3.1-8b}"
CANONICAL_PRISM_RESPONSES_DIR="${CANONICAL_PRISM_RESPONSES_DIR:-artifacts/evaluations/prism/llama3.1-8b/no_system_prompt/responses}"

if [[ -n "$USER_EVAL_DIR" ]]; then
  EVAL_DIR="$USER_EVAL_DIR"
else
  unset EVAL_DIR
fi
if [[ -n "$USER_RESPONSES_DIR" ]]; then
  RESPONSES_DIR="$USER_RESPONSES_DIR"
else
  unset RESPONSES_DIR
fi
if [[ -n "$USER_LISTWISE_PATH" ]]; then
  LISTWISE_PATH="$USER_LISTWISE_PATH"
else
  unset LISTWISE_PATH
fi
if [[ -n "$USER_PREFERENCES_PATH" ]]; then
  PREFERENCES_PATH="$USER_PREFERENCES_PATH"
else
  unset PREFERENCES_PATH
fi

safe_model_id() {
  local model_id="$1"
  printf '%s' "${model_id//\//__}" | tr ':' '_'
}

seed_missing_responses_from_canonical() {
  local canonical_dir="$CANONICAL_PRISM_RESPONSES_DIR"
  if [[ ! -d "$canonical_dir" ]]; then
    return 0
  fi

  local model_id safe_id src_path dst_path
  for model_id in "${MODEL_LIST[@]}"; do
    safe_id="$(safe_model_id "$model_id")"
    src_path="$canonical_dir/${safe_id}.jsonl"
    dst_path="$RESPONSES_DIR/${safe_id}.jsonl"
    if [[ -f "$src_path" && ! -f "$dst_path" ]]; then
      cp "$src_path" "$dst_path"
      log "Seeded cached responses for $model_id from $src_path"
    fi
  done
}

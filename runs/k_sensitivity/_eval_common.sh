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

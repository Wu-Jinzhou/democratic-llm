#!/usr/bin/env bash
set -euo pipefail

export FULL_GLOBAL_MODEL="${FULL_GLOBAL_MODEL:-${FULL_EN_GLOBAL_MODEL:-checkpoints/community-alignment/llama3.1-8b-full-global}}"
export FULL_US_MODEL="${FULL_US_MODEL:-checkpoints/community-alignment/llama3.1-8b-full-us}"
export SOFT_MODEL="${SOFT_MODEL:-checkpoints/community-alignment/llama3.1-8b-soft-en}"
export HARD_MODEL="${HARD_MODEL:-checkpoints/community-alignment/llama3.1-8b-hard-en}"

export EVAL_DIR="${EVAL_DIR:-artifacts/evaluations/community-alignment-en/llama3.1-8b/no_system_prompt}"
export LEGACY_RESPONSES_DIR="${LEGACY_RESPONSES_DIR:-artifacts/evaluations/community-alignment/llama3.1-8b/no_system_prompt/responses}"
export FULL_GLOBAL_RESPONSE_FILE="${FULL_GLOBAL_RESPONSE_FILE:-checkpoints__community-alignment__llama3.1-8b-full-global.jsonl}"

ensure_ca_en_full_global_response() {
  local responses_dir="${RESPONSES_DIR:-$EVAL_DIR/responses}"
  local target="$responses_dir/$FULL_GLOBAL_RESPONSE_FILE"
  local legacy="$LEGACY_RESPONSES_DIR/$FULL_GLOBAL_RESPONSE_FILE"
  mkdir -p "$responses_dir"
  if [[ ! -f "$target" && -f "$legacy" ]]; then
    cp "$legacy" "$target"
  fi
}

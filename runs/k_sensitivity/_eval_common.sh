#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"
# shellcheck disable=SC1091
source "$RUNS_ROOT/community_alignment/_community_alignment_eval_common.sh"

EVAL_ROOT="${EVAL_ROOT:-artifacts/evaluations/prism/k-sensitivity/llama3.1-8b}"

safe_model_id() {
  local model_id="$1"
  printf '%s' "${model_id//\//__}" | tr ':' '_'
}

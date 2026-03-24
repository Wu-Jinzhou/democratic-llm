#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_overtonbench_8b_common.sh"

log "Summarizing OvertonBench scores."
log "  SUMMARY_JSON=$SUMMARY_JSON"
log "  SUMMARY_CSV=$SUMMARY_CSV"
log "  MODELS=${MODEL_LIST[*]}"

"$PYTHON" scripts/analysis/overtonbench_eval.py summarize \
  --predictions-dir "$PREDICTIONS_DIR" \
  --models "${MODEL_LIST[@]}" \
  --summary-json "$SUMMARY_JSON" \
  --summary-csv "$SUMMARY_CSV" \
  --judge-model "$JUDGE_MODEL" \
  --prompt "$JUDGE_PROMPT"

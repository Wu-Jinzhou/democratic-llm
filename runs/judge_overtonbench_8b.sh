#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_overtonbench_8b_common.sh"

if [[ -z "${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}" ]]; then
  echo "Set GEMINI_API_KEY or GOOGLE_API_KEY before running Gemini OvertonBench judging." >&2
  exit 1
fi

log "Scoring OvertonBench rows with $JUDGE_MODEL using prompt $JUDGE_PROMPT."
log "  PREDICTIONS_DIR=$PREDICTIONS_DIR"
log "  RESPONSES_DIR=$RESPONSES_DIR"
log "  JUDGE_WORKERS=$JUDGE_WORKERS"
log "  MODELS=${MODEL_LIST[*]}"

"$PYTHON" scripts/analysis/overtonbench_eval.py judge \
  --questions-csv "$OVERTON_QUESTIONS_CSV" \
  --benchmark-csv "$OVERTON_BENCHMARK_CSV" \
  --responses-dir "$RESPONSES_DIR" \
  --predictions-dir "$PREDICTIONS_DIR" \
  --models "${MODEL_LIST[@]}" \
  --judge-model "$JUDGE_MODEL" \
  --prompt "$JUDGE_PROMPT" \
  --max-workers "$JUDGE_WORKERS"

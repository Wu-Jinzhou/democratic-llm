#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_eval_common.sh"

NUM_WORKERS="${NUM_WORKERS:-96}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"
VIS_ROOT="${VIS_ROOT:-archive/before_submission/visualization}"
DIAG_DIR="${DIAG_DIR:-$EVAL_DIR/diagnostics}"
PLOTS_DIR="${PLOTS_DIR:-$EVAL_DIR/plots}"

if [[ ! -f "$LISTWISE_PATH" ]]; then
  echo "Missing listwise results: $LISTWISE_PATH" >&2
  exit 1
fi
if [[ ! -f "$PREFERENCES_PATH" ]]; then
  echo "Missing preferences results: $PREFERENCES_PATH" >&2
  exit 1
fi
if [[ ! -d "$VIS_ROOT" ]]; then
  echo "Missing visualization scripts directory: $VIS_ROOT" >&2
  exit 1
fi

mkdir -p "$EVAL_DIR" "$DIAG_DIR" "$PLOTS_DIR"

log "PRISM soft-weighting scoring / plotting configuration:"
log "  LISTWISE_PATH=$LISTWISE_PATH"
log "  PREFERENCES_PATH=$PREFERENCES_PATH"
log "  EVAL_DIR=$EVAL_DIR"
log "  DIAG_DIR=$DIAG_DIR"
log "  PLOTS_DIR=$PLOTS_DIR"
log "  NUM_WORKERS=$NUM_WORKERS"
log "  BOOTSTRAP_SAMPLES=$BOOTSTRAP_SAMPLES"
log "  VIS_ROOT=$VIS_ROOT"

log "Scoring / ranking"
"$PYTHON" scripts/fit_bradley_terry.py \
  --preferences "$PREFERENCES_PATH" \
  --output "${EVAL_DIR}/bradley_terry_scores.json" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --bootstrap-workers "$NUM_WORKERS"

"$PYTHON" scripts/score_rankings.py \
  --listwise "$LISTWISE_PATH" \
  --output "${EVAL_DIR}/ranking_scores_plackett-luce.json" \
  --method plackett-luce \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --bootstrap-workers "$NUM_WORKERS"

"$PYTHON" scripts/score_rankings.py \
  --listwise "$LISTWISE_PATH" \
  --output "${EVAL_DIR}/ranking_scores_borda.json" \
  --method borda \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --bootstrap-workers "$NUM_WORKERS"

"$PYTHON" scripts/score_rankings.py \
  --listwise "$LISTWISE_PATH" \
  --output "${EVAL_DIR}/ranking_scores_copeland.json" \
  --method copeland \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --bootstrap-workers "$NUM_WORKERS"

"$PYTHON" scripts/score_rankings.py \
  --listwise "$LISTWISE_PATH" \
  --output "${EVAL_DIR}/ranking_scores_kemeny.json" \
  --method kemeny

"$PYTHON" scripts/score_rankings.py \
  --listwise "$LISTWISE_PATH" \
  --output "${EVAL_DIR}/ranking_scores_mallows.json" \
  --method mallows \
  --mallows-bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --mallows-bootstrap-workers "$NUM_WORKERS"

"$PYTHON" scripts/test_iia_plackett_luce.py \
  --listwise "$LISTWISE_PATH" \
  --output "${EVAL_DIR}/iia_plackett_luce.json" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --bootstrap-workers "$NUM_WORKERS"

log "Diagnostics"
"$PYTHON" scripts/analysis/judge_reliability.py \
  --listwise "$LISTWISE_PATH" \
  --output "${DIAG_DIR}/judge_reliability.json" \
  --per-question-csv "${DIAG_DIR}/judge_reliability_per_question.csv"

"$PYTHON" scripts/analysis/pairwise_effects.py \
  --preferences "$PREFERENCES_PATH" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --seed 42 \
  --output "${DIAG_DIR}/pairwise_effects.json"

"$PYTHON" scripts/ablations/ablation_order_bias.py \
  --listwise "$LISTWISE_PATH" \
  --output "${EVAL_DIR}/order_bias.json"

"$PYTHON" scripts/ablations/ablation_verbosity_bias.py \
  --listwise "$LISTWISE_PATH" \
  --output "${EVAL_DIR}/verbosity_bias.json"

log "Visualizations"
"$PYTHON" "$VIS_ROOT/plot_scores.py" \
  --input "${EVAL_DIR}/bradley_terry_scores.json" \
  --method bradley-terry \
  --output "${PLOTS_DIR}/bradley_terry_scores.png"

"$PYTHON" "$VIS_ROOT/plot_scores.py" \
  --input "${EVAL_DIR}/ranking_scores_plackett-luce.json" \
  --method plackett-luce \
  --output "${PLOTS_DIR}/plackett_luce_scores.png"

"$PYTHON" "$VIS_ROOT/plot_scores.py" \
  --input "${EVAL_DIR}/ranking_scores_borda.json" \
  --method borda \
  --output "${PLOTS_DIR}/borda_scores.png"

"$PYTHON" "$VIS_ROOT/plot_scores.py" \
  --input "${EVAL_DIR}/ranking_scores_copeland.json" \
  --method copeland \
  --output "${PLOTS_DIR}/copeland_scores.png"

"$PYTHON" "$VIS_ROOT/plot_scores.py" \
  --input "${EVAL_DIR}/ranking_scores_kemeny.json" \
  --method kemeny \
  --output "${PLOTS_DIR}/kemeny_rank_probs.png"

"$PYTHON" "$VIS_ROOT/plot_scores.py" \
  --input "${EVAL_DIR}/ranking_scores_mallows.json" \
  --method mallows \
  --output "${PLOTS_DIR}/mallows_rank_probs.png"

"$PYTHON" "$VIS_ROOT/build_clause_heatmap.py" \
  --preferences "$PREFERENCES_PATH" \
  --output-dir "$PLOTS_DIR"

"$PYTHON" "$VIS_ROOT/model_consistency.py" \
  --preferences "$PREFERENCES_PATH" \
  --output-dir "$PLOTS_DIR" \
  --plot-type box

"$PYTHON" "$VIS_ROOT/clause_difficulty.py" \
  --preferences "$PREFERENCES_PATH" \
  --output-dir "$PLOTS_DIR" \
  --metric entropy_normalized

"$PYTHON" "$VIS_ROOT/top1_win_rate.py" \
  --listwise "$LISTWISE_PATH" \
  --output-dir "$PLOTS_DIR"

"$PYTHON" "$VIS_ROOT/topk_win_rate.py" \
  --listwise "$LISTWISE_PATH" \
  --k-list 3 \
  --output-dir "$PLOTS_DIR"

"$PYTHON" "$VIS_ROOT/pairwise_winrate_heatmap.py" \
  --preferences "$PREFERENCES_PATH" \
  --output-dir "$PLOTS_DIR"

"$PYTHON" "$VIS_ROOT/rank_distribution_heatmap.py" \
  --listwise "$LISTWISE_PATH" \
  --preferences "$PREFERENCES_PATH" \
  --output-dir "$PLOTS_DIR"

"$PYTHON" "$VIS_ROOT/clause_leader_heatmap.py" \
  --preferences "$PREFERENCES_PATH" \
  --output-dir "$PLOTS_DIR"

"$PYTHON" "$VIS_ROOT/win_share_trajectory.py" \
  --preferences "$PREFERENCES_PATH" \
  --output-dir "$PLOTS_DIR"

"$PYTHON" "$VIS_ROOT/margin_distribution.py" \
  --listwise "$LISTWISE_PATH" \
  --output-dir "$PLOTS_DIR"

"$PYTHON" "$VIS_ROOT/judge_agreement.py" \
  --listwise "$LISTWISE_PATH" \
  --output-dir "$PLOTS_DIR"

"$PYTHON" "$VIS_ROOT/model_vs_baseline_delta.py" \
  --preferences "$PREFERENCES_PATH" \
  --baseline-models "$FULL_MODEL" "$HARD_MODEL" \
  --output-dir "$PLOTS_DIR" \
  --plot-name "soft_models_vs_anchors_delta.png" \
  --csv-name "soft_models_vs_anchors_delta.csv"

"$PYTHON" "$VIS_ROOT/compose_scores_grid.py" \
  --left "${PLOTS_DIR}/borda_scores.png" "${PLOTS_DIR}/kemeny_rank_probs.png" "${PLOTS_DIR}/copeland_scores.png" \
  --right "${PLOTS_DIR}/bradley_terry_scores.png" "${PLOTS_DIR}/plackett_luce_scores.png" "${PLOTS_DIR}/mallows_rank_probs.png" \
  --output "${PLOTS_DIR}/score_grid.png"

"$PYTHON" "$VIS_ROOT/pairwise_forest_grid.py" \
  --pairwise "${DIAG_DIR}/pairwise_effects.json" \
  --output "${PLOTS_DIR}/pairwise_forest_grid.png"

log "Done. Outputs written to $EVAL_DIR and $PLOTS_DIR"

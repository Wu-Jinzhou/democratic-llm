#!/usr/bin/env bash
set -euo pipefail
set -x

LISTWISE_PATH="${LISTWISE_PATH:-artifacts/evaluations/listwise.jsonl}"
PREFERENCES_PATH="${PREFERENCES_PATH:-artifacts/evaluations/preferences.jsonl}"
EVAL_DIR="${EVAL_DIR:-artifacts/evaluations}"
VIZ_DIR="${VIZ_DIR:-visualization/output}"

mkdir -p "$EVAL_DIR" "$VIZ_DIR"

python visualization/plot_scores.py \
  --input "$EVAL_DIR/bradley_terry_scores.json" \
  --method bradley-terry \
  --output "$VIZ_DIR/bradley_terry_scores.png"

python visualization/plot_scores.py \
  --input "$EVAL_DIR/ranking_scores_plackett-luce.json" \
  --method plackett-luce \
  --output "$VIZ_DIR/plackett_luce_scores.png"

python visualization/plot_scores.py \
  --input "$EVAL_DIR/ranking_scores_borda.json" \
  --method borda \
  --output "$VIZ_DIR/borda_scores.png"

python visualization/plot_scores.py \
  --input "$EVAL_DIR/ranking_scores_copeland.json" \
  --method copeland \
  --output "$VIZ_DIR/copeland_scores.png"

python visualization/plot_scores.py \
  --input "$EVAL_DIR/ranking_scores_kemeny.json" \
  --method kemeny \
  --output "$VIZ_DIR/kemeny_ranking.png"

echo "Scoring Mallows (consensus + bootstrap + pairwise)..."
python scripts/score_rankings.py \
  --listwise "$LISTWISE_PATH" \
  --output "$EVAL_DIR/ranking_scores_mallows.json" \
  --method mallows \
  --mallows-bootstrap-samples "${MALLOWS_BOOTSTRAP_SAMPLES:-500}" \
  --mallows-bootstrap-workers "${MALLOWS_BOOTSTRAP_WORKERS:-12}" \
  --mallows-pairwise-max-enum "${MALLOWS_PAIRWISE_MAX_ENUM:-8}" \
  --mallows-pairwise-mc-samples "${MALLOWS_PAIRWISE_MC_SAMPLES:-20000}" \
  --verbose

python visualization/plot_scores.py \
  --input "$EVAL_DIR/ranking_scores_mallows.json" \
  --method mallows \
  --output "$VIZ_DIR/mallows_consensus.png"

python visualization/plot_mallows_pairwise.py \
  --input "$EVAL_DIR/ranking_scores_mallows.json" \
  --output "$VIZ_DIR/mallows_pairwise.png" \
  --order consensus

echo "Building clause/model visualizations..."
python visualization/build_clause_heatmap.py \
  --preferences "$PREFERENCES_PATH" \
  --output-dir "$VIZ_DIR"

python visualization/model_consistency.py \
  --preferences "$PREFERENCES_PATH" \
  --output-dir "$VIZ_DIR" \
  --plot-type box

python visualization/clause_difficulty.py \
  --preferences "$PREFERENCES_PATH" \
  --output-dir "$VIZ_DIR" \
  --metric entropy_normalized

echo "Done. Outputs in $EVAL_DIR and $VIZ_DIR"

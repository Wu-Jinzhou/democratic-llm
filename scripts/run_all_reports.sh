#!/usr/bin/env bash
set -euo pipefail

LISTWISE_PATH="${LISTWISE_PATH:-artifacts/evaluations/listwise.jsonl}"
PREFERENCES_PATH="${PREFERENCES_PATH:-artifacts/evaluations/preferences.jsonl}"
EVAL_DIR="${EVAL_DIR:-artifacts/evaluations}"
VIZ_DIR="${VIZ_DIR:-visualization/output}"

mkdir -p "$EVAL_DIR" "$VIZ_DIR"

echo "Fitting Bradley-Terry (with bootstrap CIs)..."
python scripts/fit_bradley_terry.py \
  --preferences "$PREFERENCES_PATH" \
  --output "$EVAL_DIR/bradley_terry_scores.json" \
  --bootstrap-samples 500

python visualization/plot_scores.py \
  --input "$EVAL_DIR/bradley_terry_scores.json" \
  --method bradley-terry \
  --output "$VIZ_DIR/bradley_terry_scores.png"

echo "Scoring listwise rankings (Plackett-Luce, Borda, Copeland, Kemeny)..."
python scripts/score_rankings.py \
  --listwise "$LISTWISE_PATH" \
  --output "$EVAL_DIR/ranking_scores_plackett-luce.json" \
  --method plackett-luce \
  --bootstrap-samples 500

python visualization/plot_scores.py \
  --input "$EVAL_DIR/ranking_scores_plackett-luce.json" \
  --method plackett-luce \
  --output "$VIZ_DIR/plackett_luce_scores.png"

python scripts/score_rankings.py \
  --listwise "$LISTWISE_PATH" \
  --output "$EVAL_DIR/ranking_scores_borda.json" \
  --method borda

python visualization/plot_scores.py \
  --input "$EVAL_DIR/ranking_scores_borda.json" \
  --method borda \
  --output "$VIZ_DIR/borda_scores.png"

python scripts/score_rankings.py \
  --listwise "$LISTWISE_PATH" \
  --output "$EVAL_DIR/ranking_scores_copeland.json" \
  --method copeland

python visualization/plot_scores.py \
  --input "$EVAL_DIR/ranking_scores_copeland.json" \
  --method copeland \
  --output "$VIZ_DIR/copeland_scores.png"

python scripts/score_rankings.py \
  --listwise "$LISTWISE_PATH" \
  --output "$EVAL_DIR/ranking_scores_kemeny.json" \
  --method kemeny

python visualization/plot_scores.py \
  --input "$EVAL_DIR/ranking_scores_kemeny.json" \
  --method kemeny \
  --output "$VIZ_DIR/kemeny_ranking.png"

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

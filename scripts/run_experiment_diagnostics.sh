#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"

OUTDIR="${OUTDIR:-artifacts/evaluations/diagnostics}"
PANEL_SEED="${PANEL_SEED:-42}"
PANEL_ALGORITHM="${PANEL_ALGORITHM:-leximin}" # leximin|legacy|random
NUM_PANEL_SAMPLES="${NUM_PANEL_SAMPLES:-2000}" # used only for legacy/random soft weights
NUM_WORKERS="${NUM_WORKERS:-8}"               # used only for legacy/random soft weights
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}" # pairwise-effects bootstrap samples

mkdir -p "$OUTDIR"

echo "== Diagnostics output dir: $OUTDIR =="
echo "== Panel algorithm: $PANEL_ALGORITHM (seed=$PANEL_SEED) =="

echo "== 1) Panel feasibility + representativeness (Census config) =="
$PY scripts/analysis/panel_report.py \
  --survey prism-alignment/survey.jsonl \
  --utterances prism-alignment/utterances.jsonl \
  --panel-config configs/panel_config.yaml \
  --panel-algorithm "$PANEL_ALGORITHM" \
  --panel-seed "$PANEL_SEED" \
  --output "$OUTDIR/panel_report_census.json"

echo "== 1b) Panel feasibility + representativeness (Adversary config) =="
$PY scripts/analysis/panel_report.py \
  --survey prism-alignment/survey.jsonl \
  --utterances prism-alignment/utterances.jsonl \
  --panel-config configs/adversary_config.yaml \
  --panel-algorithm "$PANEL_ALGORITHM" \
  --panel-seed "$PANEL_SEED" \
  --output "$OUTDIR/panel_report_adversary.json"

echo "== 2) Soft panel selection-probability diagnostics (Census config) =="
$PY scripts/analysis/soft_weights_diagnostics.py \
  --survey prism-alignment/survey.jsonl \
  --utterances prism-alignment/utterances.jsonl \
  --panel-config configs/panel_config.yaml \
  --panel-algorithm "$PANEL_ALGORITHM" \
  --num-panel-samples "$NUM_PANEL_SAMPLES" \
  --panel-seed "$PANEL_SEED" \
  --num-workers "$NUM_WORKERS" \
  --dataset artifacts/data/soft_panel.jsonl \
  --per-rater-output "$OUTDIR/soft_weights_census_per_rater.csv" \
  --output "$OUTDIR/soft_weights_census.json"

echo "== 2b) Soft panel selection-probability diagnostics (Adversary config) =="
$PY scripts/analysis/soft_weights_diagnostics.py \
  --survey prism-alignment/survey.jsonl \
  --utterances prism-alignment/utterances.jsonl \
  --panel-config configs/adversary_config.yaml \
  --panel-algorithm "$PANEL_ALGORITHM" \
  --num-panel-samples "$NUM_PANEL_SAMPLES" \
  --panel-seed "$PANEL_SEED" \
  --num-workers "$NUM_WORKERS" \
  --dataset artifacts/data/adversary_soft.jsonl \
  --per-rater-output "$OUTDIR/soft_weights_adversary_per_rater.csv" \
  --output "$OUTDIR/soft_weights_adversary.json"

echo "== 3) Judge reliability (listwise, 5 judges per question) =="
$PY scripts/analysis/judge_reliability.py \
  --listwise artifacts/evaluations/listwise.jsonl \
  --output "$OUTDIR/judge_reliability.json" \
  --per-question-csv "$OUTDIR/judge_reliability_per_question.csv"

echo "== 4) Pairwise effect sizes (vote-level win rates; bootstrap) =="
$PY scripts/analysis/pairwise_effects.py \
  --preferences artifacts/evaluations/preferences.jsonl \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --seed "$PANEL_SEED" \
  --output "$OUTDIR/pairwise_effects.json"

echo "== 5) Clauses where Soft Panel beats Full PRISM (sorted) =="
$PY scripts/analysis/clauses_soft_vs_full.py \
  --preferences artifacts/evaluations/preferences.jsonl \
  --listwise artifacts/evaluations/listwise.jsonl \
  --output "$OUTDIR/clauses_soft_vs_full.csv"

echo "== Done. Outputs written under: $OUTDIR =="

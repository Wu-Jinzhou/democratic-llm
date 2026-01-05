# Visualization

This folder contains scripts for generating descriptive plots from evaluation outputs.

## Clause-by-model heatmap

Build a heatmap where rows are clauses and columns are models. Each row is normalized to sum to 1.

Input: `artifacts/evaluations/preferences.jsonl`

```bash
python visualization/build_clause_heatmap.py \
  --preferences artifacts/evaluations/preferences.jsonl \
  --output-dir visualization/output
```

Outputs:
- `visualization/output/clause_model_scores.csv` (normalized scores)
- `visualization/output/clause_model_heatmap.png` (heatmap)

Options:
- `--heatmap-name` to rename the PNG
- `--csv-name` to rename the CSV
- `--cmap` to change the matplotlib colormap (e.g., `magma`, `plasma`)

## Model consistency across clauses

This plot summarizes how stable each model is across clauses by looking at the
distribution of per-clause win shares for each model.

Method:
1. For each clause, sum a model's wins across all pairwise comparisons in that clause.
2. Normalize within the clause so each clause's model scores sum to 1.
3. For each model, collect its per-clause win shares and plot the distribution
   (box or violin).

```bash
python visualization/model_consistency.py \
  --preferences artifacts/evaluations/preferences.jsonl \
  --output-dir visualization/output \
  --plot-type box
```

Outputs:
- `visualization/output/model_consistency.csv` (long-form per-clause win shares)
- `visualization/output/model_consistency.png`

## Clause difficulty / disagreement

This plot highlights clauses with higher disagreement among models.

Method:
1. For each clause, compute normalized win shares per model (same as above).
2. Compute:
   - **Entropy** of the distribution (higher = more disagreement).
   - **Normalized entropy** in [0, 1] by dividing by `log(M)`.
   - **Variance** of the distribution (lower = more disagreement).
3. Plot the chosen metric per clause.

```bash
python visualization/clause_difficulty.py \
  --preferences artifacts/evaluations/preferences.jsonl \
  --output-dir visualization/output \
  --metric entropy_normalized
```

Outputs:
- `visualization/output/clause_difficulty.csv` (entropy/variance per clause)
- `visualization/output/clause_difficulty.png`

## Preference score plots (Bradley–Terry, Plackett–Luce, Borda, Copeland, Kemeny)

This script plots scores from the ranking/score JSON files.

Inputs:
- `artifacts/evaluations/ranking_scores.json` (from `scripts/score_rankings.py`)
- `artifacts/evaluations/bradley_terry_scores.json` (from `scripts/fit_bradley_terry.py`)

Examples:
```bash
python visualization/plot_scores.py \
  --input artifacts/evaluations/ranking_scores.json \
  --method plackett-luce \
  --output visualization/output/plackett_luce_scores.png
```

```bash
python visualization/plot_scores.py \
  --input artifacts/evaluations/ranking_scores.json \
  --method borda \
  --output visualization/output/borda_scores.png
```

```bash
python visualization/plot_scores.py \
  --input artifacts/evaluations/bradley_terry_scores.json \
  --method bradley-terry \
  --output visualization/output/bradley_terry_scores.png
```

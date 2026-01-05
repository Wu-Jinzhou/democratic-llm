#!/usr/bin/env python3
"""
Build a clause-by-model heatmap from preference JSONL.
Each clause row is normalized to sum to 1 across models.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from style import apply_style, style_axes


def load_preferences(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _clause_key(value) -> str:
    if value is None:
        return "unknown"
    return str(value)


def aggregate_wins(records: List[dict]) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    wins_by_clause: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    models = set()
    for rec in records:
        clause_id = _clause_key(rec.get("clause_id"))
        model_i = rec.get("model_i")
        model_j = rec.get("model_j")
        if model_i is None or model_j is None:
            continue
        wins_i = float(rec.get("wins_i", 0))
        wins_j = float(rec.get("wins_j", 0))
        wins_by_clause[clause_id][model_i] += wins_i
        wins_by_clause[clause_id][model_j] += wins_j
        models.add(model_i)
        models.add(model_j)
    return wins_by_clause, sorted(models)


def sort_clause_ids(ids: List[str]) -> List[str]:
    def key(value: str):
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    return sorted(ids, key=key)


def build_matrix(
    wins_by_clause: Dict[str, Dict[str, float]], models: List[str]
) -> Tuple[List[str], np.ndarray]:
    clauses = sort_clause_ids(list(wins_by_clause.keys()))
    data = np.zeros((len(clauses), len(models)), dtype=float)
    for r, clause in enumerate(clauses):
        for c, model in enumerate(models):
            data[r, c] = wins_by_clause[clause].get(model, 0.0)
    row_sums = data.sum(axis=1, keepdims=True)
    normalized = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums > 0)
    return clauses, normalized


def save_csv(
    clauses: List[str],
    models: List[str],
    matrix: np.ndarray,
    path: Path,
) -> None:
    df = pd.DataFrame(matrix, index=clauses, columns=models)
    df.index.name = "clause_id"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def save_heatmap(
    clauses: List[str],
    models: List[str],
    matrix: np.ndarray,
    path: Path,
    cmap: str,
) -> None:
    apply_style(grid=False)
    width = max(8, 0.8 * len(models))
    height = max(6, 0.25 * len(clauses))
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticks(range(len(clauses)))
    ax.set_yticklabels(clauses)
    ax.set_xlabel("Model")
    ax.set_ylabel("Clause")
    ax.set_title("Clause-by-model normalized preference scores")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    style_axes(ax, grid=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a clause-by-model heatmap from preference data.")
    parser.add_argument(
        "--preferences",
        type=Path,
        default=Path("artifacts/evaluations/preferences.jsonl"),
        help="Pairwise preference JSONL from judging.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("visualization/output"))
    parser.add_argument("--heatmap-name", default="clause_model_heatmap.png")
    parser.add_argument("--csv-name", default="clause_model_scores.csv")
    parser.add_argument("--cmap", default="viridis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_preferences(args.preferences)
    wins_by_clause, models = aggregate_wins(records)
    if not wins_by_clause:
        raise RuntimeError("No clause data found in preferences.")
    if not models:
        raise RuntimeError("No models found in preferences.")
    clauses, matrix = build_matrix(wins_by_clause, models)

    output_dir = args.output_dir
    csv_path = output_dir / args.csv_name
    heatmap_path = output_dir / args.heatmap_name
    save_csv(clauses, models, matrix, csv_path)
    save_heatmap(clauses, models, matrix, heatmap_path, args.cmap)
    print(f"Wrote CSV to {csv_path}")
    print(f"Wrote heatmap to {heatmap_path}")


if __name__ == "__main__":
    main()

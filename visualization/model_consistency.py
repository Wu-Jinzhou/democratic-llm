#!/usr/bin/env python3
"""
Model consistency across clauses.

Builds a distribution of per-clause win shares for each model and plots a box/violin chart.
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
    valid = row_sums.squeeze() > 0
    normalized = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums > 0)
    return [c for c, ok in zip(clauses, valid) if ok], normalized[valid]


def save_long_csv(
    clauses: List[str],
    models: List[str],
    matrix: np.ndarray,
    path: Path,
) -> None:
    rows = []
    for r, clause in enumerate(clauses):
        for c, model in enumerate(models):
            rows.append(
                {
                    "clause_id": clause,
                    "model": model,
                    "win_share": float(matrix[r, c]),
                }
            )
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot_distribution(
    models: List[str],
    matrix: np.ndarray,
    plot_type: str,
    path: Path,
) -> None:
    data = [matrix[:, i] for i in range(len(models))]
    width = max(8, 0.8 * len(models))
    fig, ax = plt.subplots(figsize=(width, 6))
    if plot_type == "violin":
        ax.violinplot(data, showmeans=True, showextrema=True)
        ax.set_xticks(range(1, len(models) + 1))
        ax.set_xticklabels(models, rotation=45, ha="right")
    else:
        ax.boxplot(data, labels=models, showfliers=True)
        ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Per-clause win share (row-normalized)")
    ax.set_xlabel("Model")
    ax.set_title("Model consistency across clauses")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot model consistency across clauses.")
    parser.add_argument(
        "--preferences",
        type=Path,
        default=Path("artifacts/evaluations/preferences.jsonl"),
        help="Pairwise preference JSONL from judging.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("visualization/output"))
    parser.add_argument("--plot-type", choices=["box", "violin"], default="box")
    parser.add_argument("--plot-name", default="model_consistency.png")
    parser.add_argument("--csv-name", default="model_consistency.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_preferences(args.preferences)
    wins_by_clause, models = aggregate_wins(records)
    if not wins_by_clause:
        raise RuntimeError("No clause data found in preferences.")
    clauses, matrix = build_matrix(wins_by_clause, models)
    if not clauses:
        raise RuntimeError("No clauses with comparisons found.")
    output_dir = args.output_dir
    save_long_csv(clauses, models, matrix, output_dir / args.csv_name)
    plot_distribution(models, matrix, args.plot_type, output_dir / args.plot_name)
    print(f"Wrote CSV to {output_dir / args.csv_name}")
    print(f"Wrote plot to {output_dir / args.plot_name}")


if __name__ == "__main__":
    main()

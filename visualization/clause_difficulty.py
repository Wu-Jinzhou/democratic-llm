#!/usr/bin/env python3
"""
Clause difficulty / disagreement metrics from preference data.

Computes entropy and variance of per-clause win-share distributions.
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


def compute_metrics(clauses: List[str], matrix: np.ndarray) -> pd.DataFrame:
    eps = 1e-12
    entropy = -np.sum(matrix * np.log(matrix + eps), axis=1)
    if matrix.shape[1] > 1:
        entropy_norm = entropy / np.log(matrix.shape[1])
    else:
        entropy_norm = np.zeros_like(entropy)
    variance = np.var(matrix, axis=1)
    df = pd.DataFrame(
        {
            "clause_id": clauses,
            "entropy": entropy,
            "entropy_normalized": entropy_norm,
            "variance": variance,
        }
    )
    return df


def plot_metric(
    df: pd.DataFrame,
    metric: str,
    path: Path,
    top_k: int | None,
) -> None:
    if metric not in df.columns:
        raise ValueError(f"Unknown metric: {metric}")
    plot_df = df.sort_values(metric, ascending=False)
    if top_k is not None:
        plot_df = plot_df.head(top_k)
    clauses = plot_df["clause_id"].astype(str).tolist()
    values = plot_df[metric].astype(float).tolist()
    height = max(6, 0.25 * len(clauses))
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(clauses, values)
    ax.invert_yaxis()
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("Clause")
    ax.set_title("Clause difficulty / disagreement")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute clause difficulty metrics.")
    parser.add_argument(
        "--preferences",
        type=Path,
        default=Path("artifacts/evaluations/preferences.jsonl"),
        help="Pairwise preference JSONL from judging.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("visualization/output"))
    parser.add_argument("--csv-name", default="clause_difficulty.csv")
    parser.add_argument("--plot-name", default="clause_difficulty.png")
    parser.add_argument(
        "--metric",
        choices=["entropy", "entropy_normalized", "variance"],
        default="entropy_normalized",
        help="Metric to plot.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Optional top-k clauses to plot.")
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

    df = compute_metrics(clauses, matrix)
    output_dir = args.output_dir
    csv_path = output_dir / args.csv_name
    plot_path = output_dir / args.plot_name
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    plot_metric(df, args.metric, plot_path, args.top_k)
    print(f"Wrote CSV to {csv_path}")
    print(f"Wrote plot to {plot_path}")


if __name__ == "__main__":
    main()

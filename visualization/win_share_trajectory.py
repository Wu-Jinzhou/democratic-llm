#!/usr/bin/env python3
"""
Plot per-clause win-share trajectory for each model.
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

from style import apply_style, style_axes, single_hue_palette, display_model_names


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


def sort_clause_ids(ids: List[str]) -> List[str]:
    def key(value: str):
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    return sorted(ids, key=key)


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


def save_csv(clauses: List[str], models: List[str], matrix: np.ndarray, path: Path) -> None:
    df = pd.DataFrame(matrix, index=clauses, columns=models)
    df.index.name = "clause_id"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def plot(clauses: List[str], models: List[str], matrix: np.ndarray, path: Path) -> None:
    apply_style(grid=False)
    colors = single_hue_palette(len(models), cmap_name="Blues", start=0.45, end=0.9)
    x = list(range(len(clauses)))
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = display_model_names(models)
    for idx, label in enumerate(labels):
        ax.plot(x, matrix[:, idx], color=colors[idx], linewidth=1.6, label=label)
    ax.set_xlabel("Clause index")
    ax.set_ylabel("Win share (row-normalized)")
    ax.set_title("Win-share trajectory by clause")
    ax.set_xticks(x[:: max(1, len(x) // 10)])
    ax.set_xticklabels([clauses[i] for i in ax.get_xticks().astype(int)])
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)
    style_axes(ax, grid=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot win-share trajectory by clause.")
    parser.add_argument(
        "--preferences",
        type=Path,
        default=Path("artifacts/evaluations/preferences.jsonl"),
        help="Pairwise preference JSONL.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("visualization/output"))
    parser.add_argument("--plot-name", default="win_share_trajectory.png")
    parser.add_argument("--csv-name", default="win_share_trajectory.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_preferences(args.preferences)
    wins_by_clause, models = aggregate_wins(records)
    if not wins_by_clause:
        raise RuntimeError("No clause data found in preferences.")
    clauses, matrix = build_matrix(wins_by_clause, models)
    save_csv(clauses, models, matrix, args.output_dir / args.csv_name)
    plot(clauses, models, matrix, args.output_dir / args.plot_name)
    print(f"Wrote plot to {args.output_dir / args.plot_name}")


if __name__ == "__main__":
    main()

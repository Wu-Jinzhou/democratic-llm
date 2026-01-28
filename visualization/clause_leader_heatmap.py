#!/usr/bin/env python3
"""
Plot the top model per clause as a categorical heatmap.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

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


def build_leader(wins_by_clause: Dict[str, Dict[str, float]], models: List[str]) -> Tuple[List[str], List[str]]:
    clauses = sort_clause_ids(list(wins_by_clause.keys()))
    leaders = []
    for clause in clauses:
        wins = wins_by_clause[clause]
        best = max(models, key=lambda m: wins.get(m, 0.0))
        leaders.append(best)
    return clauses, leaders


def save_csv(clauses: List[str], leaders: List[str], path: Path) -> None:
    df = pd.DataFrame({"clause_id": clauses, "leader_model": leaders})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot(clauses: List[str], leaders: List[str], models: List[str], path: Path) -> None:
    apply_style(grid=False)
    model_to_idx = {m: i for i, m in enumerate(models)}
    matrix = np.array([[model_to_idx[m]] for m in leaders], dtype=float)
    colors = single_hue_palette(len(models), cmap_name="Blues", start=0.45, end=0.9)
    cmap = mcolors.ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(6, max(6, 0.25 * len(clauses))))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=len(models) - 1)
    ax.set_yticks(range(len(clauses)))
    ax.set_yticklabels(clauses)
    ax.set_xticks([0])
    ax.set_xticklabels(["Leader"])
    ax.set_ylabel("Clause")
    ax.set_title("Top model per clause")
    ax.tick_params(axis="y", labelsize=8, length=0)
    style_axes(ax, grid=False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(models))]
    ax.legend(
        handles,
        display_model_names(models),
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot clause-level leader heatmap.")
    parser.add_argument(
        "--preferences",
        type=Path,
        default=Path("artifacts/evaluations/preferences.jsonl"),
        help="Pairwise preference JSONL.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("visualization/output"))
    parser.add_argument("--plot-name", default="clause_leader.png")
    parser.add_argument("--csv-name", default="clause_leader.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_preferences(args.preferences)
    wins_by_clause, models = aggregate_wins(records)
    if not wins_by_clause:
        raise RuntimeError("No clause data found in preferences.")
    clauses, leaders = build_leader(wins_by_clause, models)
    save_csv(clauses, leaders, args.output_dir / args.csv_name)
    plot(clauses, leaders, models, args.output_dir / args.plot_name)
    print(f"Wrote plot to {args.output_dir / args.plot_name}")


if __name__ == "__main__":
    main()

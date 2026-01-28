#!/usr/bin/env python3
"""
Plot per-clause win-share deltas vs baseline models.
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
import matplotlib.colors as mcolors

from style import apply_style, style_axes, display_model_names, truncated_cmap


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
    vmax = np.nanmax(np.abs(matrix)) if matrix.size else 1.0
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    cmap = truncated_cmap("Blues", minval=0.35, maxval=0.95)
    fig, ax = plt.subplots(figsize=(10, max(6, 0.25 * len(clauses))))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(display_model_names(models), rotation=45, ha="right")
    ax.set_yticks(range(len(clauses)))
    ax.set_yticklabels(clauses)
    ax.set_xlabel("Model")
    ax.set_ylabel("Clause")
    ax.set_title("Win-share delta vs baseline (row-normalized)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Delta win share")
    ax.tick_params(axis="x", labelsize=9, length=0)
    ax.tick_params(axis="y", labelsize=8, length=0)
    style_axes(ax, grid=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot model vs baseline win-share deltas.")
    parser.add_argument(
        "--preferences",
        type=Path,
        default=Path("artifacts/evaluations/preferences.jsonl"),
        help="Pairwise preference JSONL.",
    )
    parser.add_argument("--baseline-models", nargs="+", required=True)
    parser.add_argument("--compare-models", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("visualization/output"))
    parser.add_argument("--plot-name", default="model_vs_baseline_delta.png")
    parser.add_argument("--csv-name", default="model_vs_baseline_delta.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_preferences(args.preferences)
    wins_by_clause, models = aggregate_wins(records)
    if not wins_by_clause:
        raise RuntimeError("No clause data found in preferences.")
    clauses, matrix = build_matrix(wins_by_clause, models)

    baseline = args.baseline_models
    baseline_idx = [models.index(m) for m in baseline if m in models]
    if not baseline_idx:
        raise ValueError("No baseline models found in data.")

    compare_models = args.compare_models
    if compare_models is None:
        compare_models = [m for m in models if m not in baseline]

    compare_idx = [models.index(m) for m in compare_models if m in models]
    if not compare_idx:
        raise ValueError("No comparison models found in data.")

    baseline_avg = np.mean(matrix[:, baseline_idx], axis=1, keepdims=True)
    delta = matrix[:, compare_idx] - baseline_avg

    save_csv(clauses, [models[i] for i in compare_idx], delta, args.output_dir / args.csv_name)
    plot(clauses, [models[i] for i in compare_idx], delta, args.output_dir / args.plot_name)
    print(f"Wrote plot to {args.output_dir / args.plot_name}")


if __name__ == "__main__":
    main()

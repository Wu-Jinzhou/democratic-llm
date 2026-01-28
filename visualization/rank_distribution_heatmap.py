#!/usr/bin/env python3
"""
Plot rank distribution per model as a heatmap.
Each cell is P(model is at rank r).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

from style import apply_style, style_axes, display_model_names, display_order_index, truncated_cmap


def load_listwise(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def compute_distribution(records: List[dict]) -> Tuple[List[str], np.ndarray]:
    models = set()
    rank_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    total = 0
    max_rank = 0
    for rec in records:
        for ranking in rec.get("rankings", []):
            if not ranking:
                continue
            total += 1
            for idx, model in enumerate(ranking, start=1):
                rank_counts[model][idx] += 1
                models.add(model)
                max_rank = max(max_rank, idx)
    if total == 0:
        raise RuntimeError("No rankings found.")
    models = sorted(models)
    matrix = np.zeros((len(models), max_rank), dtype=float)
    for i, model in enumerate(models):
        for r in range(1, max_rank + 1):
            matrix[i, r - 1] = rank_counts[model].get(r, 0) / total
    return models, matrix


def compute_total_wins(preferences_path: Path) -> Dict[str, float] | None:
    if not preferences_path.exists():
        return None
    wins: Dict[str, float] = defaultdict(float)
    with preferences_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            mi = rec.get("model_i")
            mj = rec.get("model_j")
            if mi is None or mj is None:
                continue
            wins[mi] += float(rec.get("wins_i", 0.0))
            wins[mj] += float(rec.get("wins_j", 0.0))
    return wins or None


def sort_models(models: List[str], total_wins: Dict[str, float] | None) -> List[str]:
    if total_wins:
        return sorted(models, key=lambda m: total_wins.get(m, 0.0), reverse=True)
    return sorted(models, key=display_order_index)


def save_csv(models: List[str], matrix: np.ndarray, path: Path) -> None:
    cols = [f"rank_{i}" for i in range(1, matrix.shape[1] + 1)]
    df = pd.DataFrame(matrix, index=models, columns=cols)
    df.index.name = "model"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def plot(models: List[str], matrix: np.ndarray, path: Path) -> None:
    apply_style(grid=False)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(models))))
    vmax = float(np.max(matrix)) if matrix.size else 1.0
    norm = mcolors.PowerNorm(gamma=0.6, vmin=0.0, vmax=max(vmax, 1e-8))
    cmap = truncated_cmap("Blues", minval=0.35, maxval=0.95)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels([str(i) for i in range(1, matrix.shape[1] + 1)])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(display_model_names(models))
    ax.set_xlabel("Rank position")
    ax.set_ylabel("Model")
    ax.set_title("Rank distribution by model")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Probability")
    style_axes(ax, grid=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot rank distribution heatmap.")
    parser.add_argument(
        "--listwise",
        type=Path,
        default=Path("artifacts/evaluations/listwise.jsonl"),
        help="Listwise rankings JSONL.",
    )
    parser.add_argument(
        "--preferences",
        type=Path,
        default=Path("artifacts/evaluations/preferences.jsonl"),
        help="Preference JSONL used to order models by total wins.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("visualization/output"))
    parser.add_argument("--plot-name", default="rank_distribution.png")
    parser.add_argument("--csv-name", default="rank_distribution.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_listwise(args.listwise)
    models, matrix = compute_distribution(records)
    total_wins = compute_total_wins(args.preferences)
    order = sort_models(models, total_wins)
    order_idx = [models.index(m) for m in order]
    matrix = matrix[order_idx, :]
    save_csv(order, matrix, args.output_dir / args.csv_name)
    plot(order, matrix, args.output_dir / args.plot_name)
    print(f"Wrote plot to {args.output_dir / args.plot_name}")


if __name__ == "__main__":
    main()

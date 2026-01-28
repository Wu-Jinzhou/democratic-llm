#!/usr/bin/env python3
"""
Plot pairwise win-rate heatmap from preference data.
Each cell (i, j) is P(i wins over j).
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


def aggregate(records: List[dict]) -> Tuple[List[str], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    wins = defaultdict(float)
    totals = defaultdict(float)
    models = set()
    for rec in records:
        mi = rec.get("model_i")
        mj = rec.get("model_j")
        if mi is None or mj is None:
            continue
        wi = float(rec.get("wins_i", 0.0))
        wj = float(rec.get("wins_j", 0.0))
        wins[(mi, mj)] += wi
        wins[(mj, mi)] += wj
        totals[(mi, mj)] += wi + wj
        totals[(mj, mi)] += wi + wj
        models.add(mi)
        models.add(mj)
    return sorted(models), wins, totals


def build_matrix(models: List[str], wins: Dict[Tuple[str, str], float], totals: Dict[Tuple[str, str], float]) -> np.ndarray:
    n = len(models)
    matrix = np.full((n, n), np.nan, dtype=float)
    for i, mi in enumerate(models):
        for j, mj in enumerate(models):
            if i == j:
                matrix[i, j] = 0.5
                continue
            total = totals.get((mi, mj), 0.0)
            if total > 0:
                matrix[i, j] = wins.get((mi, mj), 0.0) / total
    return matrix


def save_csv(models: List[str], matrix: np.ndarray, path: Path) -> None:
    df = pd.DataFrame(matrix, index=models, columns=models)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def plot(models: List[str], matrix: np.ndarray, path: Path, annotate: bool) -> None:
    apply_style(grid=False)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(models))))

    # Diagonal is not meaningful for pairwise comparisons; hide it so autoscaling
    # uses only real pairwise win rates.
    display = matrix.copy()
    np.fill_diagonal(display, np.nan)

    cmap = truncated_cmap("Blues", minval=0.35, maxval=0.95).copy()
    cmap.set_bad(color="#F2F2F2")

    # Autoscale the colormap to the observed range so subtle deviations around 0.5
    # remain visually distinguishable (win rates are typically close to 0.5).
    finite = display[np.isfinite(display)]
    if finite.size:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    else:
        vmin, vmax = 0.0, 1.0
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(display, cmap=cmap, norm=norm)
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(models)))
    labels = display_model_names(models)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Opponent model")
    ax.set_ylabel("Row model")
    ax.set_title(f"Pairwise win rates P(row wins over col) (scaled to [{vmin:.3f}, {vmax:.3f}])")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Win rate (autoscaled)")
    style_axes(ax, grid=False)

    if annotate:
        for i in range(len(models)):
            for j in range(len(models)):
                val = display[i, j]
                if np.isnan(val):
                    continue
                # Choose annotation color based on normalized intensity, not absolute 0..1 scale.
                intensity = float(norm(val))
                color = "white" if intensity >= 0.65 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot pairwise win-rate heatmap.")
    parser.add_argument(
        "--preferences",
        type=Path,
        default=Path("artifacts/evaluations/preferences.jsonl"),
        help="Pairwise preference JSONL.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("visualization/output"))
    parser.add_argument("--plot-name", default="pairwise_winrate.png")
    parser.add_argument("--csv-name", default="pairwise_winrate.csv")
    parser.add_argument("--annotate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_preferences(args.preferences)
    models, wins, totals = aggregate(records)
    if not models:
        raise RuntimeError("No models found in preferences.")
    matrix = build_matrix(models, wins, totals)
    save_csv(models, matrix, args.output_dir / args.csv_name)
    plot(models, matrix, args.output_dir / args.plot_name, args.annotate)
    print(f"Wrote plot to {args.output_dir / args.plot_name}")


if __name__ == "__main__":
    main()

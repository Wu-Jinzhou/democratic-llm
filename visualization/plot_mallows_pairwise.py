#!/usr/bin/env python3
"""
Plot Mallows pairwise win probabilities as a heatmap.

Expects a JSON produced by scripts/score_rankings.py with mallows.pairwise_probabilities.
Each cell (i, j) is P(i beats j) under the Mallows model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from style import apply_style, style_axes, display_model_names, truncated_cmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Mallows pairwise win probability heatmap.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/evaluations/ranking_scores_mallows.json"),
        help="Mallows ranking JSON from scripts/score_rankings.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("visualization/output/mallows_pairwise.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--order",
        choices=["consensus", "models", "alpha"],
        default="consensus",
        help="Row/column order for models.",
    )
    parser.add_argument("--cmap", default="Blues")
    parser.add_argument("--annotate", action="store_true", help="Write values inside cells.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_style(grid=False)
    data = json.loads(args.input.read_text())
    mallows = data.get("mallows", {})
    pairwise = mallows.get("pairwise_probabilities", {})
    matrix = pairwise.get("matrix")
    if not isinstance(matrix, dict):
        raise ValueError("Mallows pairwise_probabilities.matrix not found.")

    if args.order == "consensus" and isinstance(mallows.get("consensus"), list):
        models: List[str] = list(mallows["consensus"])
    elif args.order == "models" and isinstance(data.get("models"), list):
        models = list(data["models"])
    else:
        models = sorted(matrix.keys())

    n = len(models)
    values = np.zeros((n, n), dtype=float)
    for i, mi in enumerate(models):
        for j, mj in enumerate(models):
            if i == j:
                values[i, j] = 0.5
                continue
            values[i, j] = float(matrix.get(mi, {}).get(mj, np.nan))

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * n)))
    norm = mcolors.TwoSlopeNorm(vcenter=0.5, vmin=0.0, vmax=1.0)
    cmap = truncated_cmap(args.cmap, minval=0.35, maxval=0.95)
    im = ax.imshow(values, cmap=cmap, norm=norm)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = display_model_names(models)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Opponent model")
    ax.set_ylabel("Row model")
    title = "Mallows pairwise win probability P(row beats col)"
    details = []
    phi = mallows.get("phi")
    if isinstance(phi, (int, float)):
        details.append(f"phi={phi:.3f}")
    if details:
        title += f" ({', '.join(details)})"
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("P(row beats col)")
    style_axes(ax, grid=False)

    if args.annotate:
        for i in range(n):
            for j in range(n):
                val = values[i, j]
                text_color = "white" if val >= 0.7 or val <= 0.3 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=text_color)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"Wrote plot to {args.output}")


if __name__ == "__main__":
    main()

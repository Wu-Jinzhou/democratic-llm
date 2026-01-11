#!/usr/bin/env python3
"""
Shared plotting style for publication-ready figures.
"""
from __future__ import annotations

import matplotlib as mpl


_CATEGORY_PALETTE = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#EECA3B",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
]


def apply_style(grid: bool = False) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#222222",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "grid.color": "#D0D0D0",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.5,
            "axes.grid": grid,
            "legend.frameon": False,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.prop_cycle": mpl.cycler(color=_CATEGORY_PALETTE),
        }
    )


def style_axes(ax, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, axis="y", alpha=0.4)
    else:
        ax.grid(False)
    ax.tick_params(axis="both", which="both", length=3, color="#222222")


def categorical_palette(n: int) -> list[str]:
    if n <= 0:
        return []
    return [_CATEGORY_PALETTE[i % len(_CATEGORY_PALETTE)] for i in range(n)]

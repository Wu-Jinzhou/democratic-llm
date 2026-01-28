#!/usr/bin/env python3
"""
Shared plotting style for publication-ready figures.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np


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

_DISPLAY_ORDER = [
    "Soft Panel",
    "Hard Panel",
    "Full PRISM",
    "US-Rep",
    "Adversarial Soft",
    "Adversarial Hard",
    "Base",
]


def display_model_name(model: str) -> str:
    text = str(model)
    lower = text.lower()
    if "adversary-soft" in lower:
        return "Adversarial Soft"
    if "adversary-hard" in lower:
        return "Adversarial Hard"
    if "soft-panel" in lower:
        return "Soft Panel"
    if "hard-panel" in lower:
        return "Hard Panel"
    if "full-prism" in lower:
        return "Full PRISM"
    if "us-rep" in lower or "us_rep" in lower:
        return "US-Rep"
    if "meta-llama" in lower and "llama-3.1-8b" in lower:
        return "Base"
    if "llama-3.1-8b" in lower and "checkpoints" not in lower:
        return "Base"
    return text


def display_model_names(models: list[str]) -> list[str]:
    return [display_model_name(model) for model in models]


def display_order_index(model: str) -> int:
    name = display_model_name(model)
    if name in _DISPLAY_ORDER:
        return _DISPLAY_ORDER.index(name)
    return len(_DISPLAY_ORDER)


def apply_style(grid: bool = False) -> None:
    mpl.rcParams.update(
        {
            # Default text in paper-style serif.
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            # Titles in Roboto (falls back cleanly if Roboto is unavailable).
            "font.sans-serif": ["Roboto", "Arial", "DejaVu Sans"],
            "font.size": 16,
            # Larger title size, but keep other text unchanged.
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
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
            "figure.dpi": 500,
            "savefig.dpi": 500,
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

    # Titles: Roboto + larger font (only affects titles).
    title = getattr(ax, "title", None)
    if title is not None and title.get_text():
        title.set_fontfamily("sans-serif")
        # Ensure we only scale title text.
        title.set_fontsize(mpl.rcParams.get("axes.titlesize", 16))
        title.set_fontweight(mpl.rcParams.get("axes.titleweight", "bold"))


def categorical_palette(n: int) -> list[str]:
    if n <= 0:
        return []
    return [_CATEGORY_PALETTE[i % len(_CATEGORY_PALETTE)] for i in range(n)]


def single_hue_palette(
    n: int,
    cmap_name: str = "Blues",
    start: float = 0.4,
    end: float = 0.9,
) -> list[str]:
    if n <= 0:
        return []
    cmap = cm.get_cmap(cmap_name)
    values = np.linspace(start, end, n)
    return [mpl.colors.to_hex(cmap(v)) for v in values]


def truncated_cmap(
    cmap_name: str = "Blues",
    minval: float = 0.35,
    maxval: float = 0.95,
    n: int = 256,
) -> mcolors.LinearSegmentedColormap:
    base = cm.get_cmap(cmap_name)
    colors = base(np.linspace(minval, maxval, n))
    return mcolors.LinearSegmentedColormap.from_list(f"{cmap_name}_trunc", colors)

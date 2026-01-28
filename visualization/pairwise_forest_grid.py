#!/usr/bin/env python3
"""
Forest-plot grid of pairwise effect sizes from bootstrap win-rate estimates.

Each panel uses one model as the *baseline* and plots, for every other model m:

    Δ(m vs baseline) = P(m beats baseline) - 0.5

with optional bootstrap confidence intervals.

This is useful when win rates are all near 0.5: the plot zooms into differences
around 0 and makes comparisons visually legible.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from style import apply_style, display_model_name, display_order_index, style_axes, truncated_cmap


@dataclass(frozen=True)
class PairEstimate:
    model_a: str
    model_b: str
    win_rate_a: float
    ci_lower_a: Optional[float]
    ci_upper_a: Optional[float]


# For this figure, we swap the relative ordering of US-Rep and Adversarial Soft
# to group the two adversarial variants together.
_PAIRWISE_DISPLAY_ORDER = [
    "Soft Panel",
    "Hard Panel",
    "Full PRISM",
    "Adversarial Soft",
    "US-Rep",
    "Adversarial Hard",
    "Base",
]


def _pairwise_order_index(model: str) -> int:
    name = display_model_name(model)
    if name in _PAIRWISE_DISPLAY_ORDER:
        return _PAIRWISE_DISPLAY_ORDER.index(name)
    # Fall back to the global display order for unknown models.
    return display_order_index(model)


def load_pairwise_estimates(path: Path) -> Tuple[List[str], List[PairEstimate], Optional[int]]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get("method") != "pairwise-effects":
        raise ValueError("Expected a pairwise-effects JSON from scripts/analysis/pairwise_effects.py")
    pairs_raw = data.get("pairs")
    if not isinstance(pairs_raw, list):
        raise ValueError("Missing 'pairs' list.")
    models: List[str] = []
    model_set = set()
    pairs: List[PairEstimate] = []
    for rec in pairs_raw:
        a = str(rec["model_a"])
        b = str(rec["model_b"])
        model_set.add(a)
        model_set.add(b)
        pairs.append(
            PairEstimate(
                model_a=a,
                model_b=b,
                win_rate_a=float(rec.get("win_rate_a_votes", rec.get("win_rate_a", 0.0))),
                ci_lower_a=(float(rec["ci_lower"]) if rec.get("ci_lower") is not None else None),
                ci_upper_a=(float(rec["ci_upper"]) if rec.get("ci_upper") is not None else None),
            )
        )
    models = sorted(model_set, key=_pairwise_order_index)
    bootstrap_n = data.get("bootstrap_samples")
    bootstrap_n_int = int(bootstrap_n) if isinstance(bootstrap_n, int) and bootstrap_n > 0 else None
    return models, pairs, bootstrap_n_int


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def build_lookup(pairs: List[PairEstimate]) -> Dict[Tuple[str, str], PairEstimate]:
    out: Dict[Tuple[str, str], PairEstimate] = {}
    for p in pairs:
        out[(p.model_a, p.model_b)] = p
    return out


def win_rate_for(m: str, baseline: str, lookup: Dict[Tuple[str, str], PairEstimate]) -> Tuple[float, Optional[float], Optional[float]]:
    a, b = _pair_key(m, baseline)
    est = lookup.get((a, b))
    if est is None:
        raise KeyError(f"Missing pair estimate for ({m}, {baseline})")
    if m == est.model_a:
        return est.win_rate_a, est.ci_lower_a, est.ci_upper_a
    # Symmetry: P(m beats b) = 1 - P(b beats m)
    wr = 1.0 - est.win_rate_a
    lo = hi = None
    if est.ci_lower_a is not None and est.ci_upper_a is not None:
        lo = 1.0 - est.ci_upper_a
        hi = 1.0 - est.ci_lower_a
    return wr, lo, hi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forest-plot grid of pairwise effect sizes.")
    parser.add_argument(
        "--pairwise",
        type=Path,
        default=Path("artifacts/evaluations/diagnostics/pairwise_effects.json"),
        help="Pairwise effects JSON (from scripts/analysis/pairwise_effects.py).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("visualization/output/pairwise_forest_grid.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_style(grid=False)

    models, pairs, bootstrap_n = load_pairwise_estimates(args.pairwise)
    lookup = build_lookup(pairs)

    # Use consistent ordering for y-axis across all panels.
    ordered = sorted(models, key=_pairwise_order_index)

    # Precompute all deltas to set a shared x-scale.
    deltas = []
    for baseline in ordered:
        for m in ordered:
            if m == baseline:
                continue
            wr, _lo, _hi = win_rate_for(m, baseline, lookup)
            deltas.append(wr - 0.5)
    max_abs = float(np.max(np.abs(deltas))) if deltas else 0.05
    xlim = max(0.05, max_abs + 0.01)

    n = len(ordered)
    fig_w = max(12, 2.0 * n)
    fig_h = max(6, 0.55 * n + 2.0)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), sharey=True)
    if n == 1:
        axes = [axes]  # type: ignore

    cmap = truncated_cmap("Blues", minval=0.35, maxval=0.95)

    y_all = list(range(len(ordered)))
    y_labels = [display_model_name(m) for m in ordered]

    for col, baseline in enumerate(ordered):
        ax = axes[col]
        ys = []
        xs = []
        xerr_low = []
        xerr_high = []
        colors = []
        for y, m in enumerate(ordered):
            if m == baseline:
                continue
            wr, lo, hi = win_rate_for(m, baseline, lookup)
            delta = wr - 0.5
            ys.append(y)
            xs.append(delta)
            if lo is not None and hi is not None:
                xerr_low.append(delta - (lo - 0.5))
                xerr_high.append((hi - 0.5) - delta)
            else:
                xerr_low.append(0.0)
                xerr_high.append(0.0)
            # Color by effect direction/magnitude (single-hue palette; darker = larger |Δ|).
            intensity = min(1.0, abs(delta) / xlim) if xlim > 0 else 0.0
            colors.append(cmap(0.35 + 0.6 * intensity))

        ax.axvline(0.0, color="#222222", linewidth=0.9, zorder=0)
        ax.errorbar(
            xs,
            ys,
            xerr=[xerr_low, xerr_high],
            fmt="o",
            markersize=7.5,
            color="#1f77b4",
            ecolor="#222222",
            elinewidth=1.6,
            capsize=3.5,
            capthick=1.6,
            zorder=2,
        )
        # Recolor markers to encode magnitude without changing errorbar color.
        for x, y, c in zip(xs, ys, colors):
            ax.scatter([x], [y], s=70, color=c, edgecolor="none", zorder=3)

        ax.set_xlim(-xlim, xlim)
        ax.set_xticks([-xlim, -xlim / 2, 0.0, xlim / 2, xlim])
        ax.set_xticklabels([f"{t:+.2f}" for t in [-xlim, -xlim / 2, 0.0, xlim / 2, xlim]])

        ax.set_title(display_model_name(baseline))
        ax.set_xlabel("Δ win rate")

        ax.set_yticks(y_all)
        if col == 0:
            ax.set_yticklabels(y_labels)
            ax.tick_params(axis="y", labelleft=True)
            ax.set_ylabel("Model")
        else:
            # sharey=True links the y-axis formatter across panels; do not mutate
            # tick labels here (it would erase them everywhere). Just hide them.
            ax.tick_params(axis="y", labelleft=False)
        ax.invert_yaxis()
        style_axes(ax, grid=False)
        # Panel headings are model names; keep them in the paper's serif font
        # and not bold (only the main figure title should be bold).
        ax.title.set_fontfamily("serif")
        ax.title.set_fontweight("normal")

    main_title = args.title
    if main_title is None:
        main_title = "Pairwise effect sizes vs each baseline"
    st = fig.suptitle(main_title, y=1.02, fontsize=18, fontfamily="sans-serif", fontweight="bold")
    st.set_fontweight("bold")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    plt.close(fig)
    print(f"Wrote plot to {args.output}")


if __name__ == "__main__":
    main()

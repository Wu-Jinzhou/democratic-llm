#!/usr/bin/env python3
"""
Plot top-k win rate per model from listwise rankings.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
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


def compute_topk(records: List[dict], k_list: List[int]) -> Dict[str, Dict[int, float]]:
    """Compute exact rank rates: P(rank == k) for each k in k_list."""
    counts = defaultdict(lambda: {k: 0 for k in k_list})
    total = 0
    for rec in records:
        for ranking in rec.get("rankings", []):
            if not ranking:
                continue
            total += 1
            for idx, model in enumerate(ranking, start=1):
                if idx in counts[model]:
                    counts[model][idx] += 1
    if total == 0:
        raise RuntimeError("No rankings found.")
    return {model: {k: counts[model][k] / total for k in k_list} for model in counts}


def sort_models_by_sum(data: Dict[str, Dict[int, float]], k_list: List[int]) -> List[str]:
    models = list(data.keys())
    return sorted(
        models,
        key=lambda m: (-sum(data[m].get(k, 0.0) for k in k_list), display_order_index(m)),
    )


def save_csv(
    data: Dict[str, Dict[int, float]], k_list: List[int], order: List[str], path: Path
) -> None:
    rows = []
    for model in order:
        metrics = data.get(model, {})
        row = {"model": model}
        for k in k_list:
            row[f"top{k}_win_rate"] = metrics.get(k, 0.0)
        rows.append(row)
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot(
    data: Dict[str, Dict[int, float]], k_list: List[int], order: List[str], path: Path
) -> None:
    models = order
    apply_style(grid=False)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(models))))
    cmap = truncated_cmap("Blues", minval=0.35, maxval=0.95)
    color_steps = np.linspace(0.35, 0.9, len(k_list))
    colors = [cmap(step) for step in color_steps]

    y = np.arange(len(models))
    left = np.zeros(len(models))
    for idx, k in enumerate(k_list):
        segment = [data.get(model, {}).get(k, 0.0) for model in models]
        ax.barh(y, segment, left=left, color=colors[idx], edgecolor="none", label=f"Rank {k}")
        left = [l + s for l, s in zip(left, segment)]

    ax.set_yticks(y)
    ax.set_yticklabels(display_model_names(models))
    ax.invert_yaxis()
    ax.set_xlabel("Win rate")
    ax.set_title("Top-k win rates by model")
    ax.legend(loc="lower right", frameon=False)
    style_axes(ax, grid=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot top-k win rates per model.")
    parser.add_argument(
        "--listwise",
        type=Path,
        default=Path("artifacts/evaluations/listwise.jsonl"),
        help="Listwise rankings JSONL.",
    )
    parser.add_argument(
        "--k-list",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Top-k cutoffs to plot.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("visualization/output"))
    parser.add_argument("--plot-name", default="topk_win_rate.png")
    parser.add_argument("--csv-name", default="topk_win_rate.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_k_list = sorted(set(args.k_list))
    if not plot_k_list:
        raise ValueError("k-list must contain at least one value.")
    order_k_list = list(range(1, max(plot_k_list) + 1))
    records = load_listwise(args.listwise)
    data_full = compute_topk(records, order_k_list)
    data_plot = {
        model: {k: data_full.get(model, {}).get(k, 0.0) for k in plot_k_list}
        for model in data_full
    }
    order = sort_models_by_sum(data_full, order_k_list)
    save_csv(data_plot, plot_k_list, order, args.output_dir / args.csv_name)
    plot(data_plot, plot_k_list, order, args.output_dir / args.plot_name)
    print(f"Wrote plot to {args.output_dir / args.plot_name}")


if __name__ == "__main__":
    main()

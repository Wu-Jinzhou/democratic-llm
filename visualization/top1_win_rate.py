#!/usr/bin/env python3
"""
Plot top-1 win rate per model from listwise rankings.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors

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


def compute_top1(records: List[dict]) -> Dict[str, float]:
    counts = defaultdict(int)
    total = 0
    for rec in records:
        for ranking in rec.get("rankings", []):
            if not ranking:
                continue
            counts[ranking[0]] += 1
            total += 1
    if total == 0:
        raise RuntimeError("No rankings found.")
    return {model: counts[model] / total for model in counts}


def sort_models_by_rate(data: Dict[str, float]) -> List[str]:
    models = list(data.keys())
    return sorted(models, key=lambda m: (-data.get(m, 0.0), display_order_index(m)))


def save_csv(data: Dict[str, float], order: List[str], path: Path) -> None:
    df = pd.DataFrame(
        [{"model": model, "top1_win_rate": data.get(model, 0.0)} for model in order]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot(data: Dict[str, float], order: List[str], path: Path) -> None:
    models = order
    values = [data.get(m, 0.0) for m in models]
    apply_style(grid=False)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(models))))
    y = list(range(len(models)))
    vmin = min(values) if values else 0.0
    vmax = max(values) if values else 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-8))
    cmap = truncated_cmap("Blues", minval=0.35, maxval=0.95)
    colors = cmap(norm(values))
    ax.barh(y, values, color=colors, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(display_model_names(models))
    ax.invert_yaxis()
    ax.set_xlabel("Top-1 win rate")
    ax.set_title("Top-1 win rate by model")
    style_axes(ax, grid=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot top-1 win rate per model.")
    parser.add_argument(
        "--listwise",
        type=Path,
        default=Path("artifacts/evaluations/listwise.jsonl"),
        help="Listwise rankings JSONL.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("visualization/output"))
    parser.add_argument("--plot-name", default="top1_win_rate.png")
    parser.add_argument("--csv-name", default="top1_win_rate.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_listwise(args.listwise)
    data = compute_top1(records)
    order = sort_models_by_rate(data)
    save_csv(data, order, args.output_dir / args.csv_name)
    plot(data, order, args.output_dir / args.plot_name)
    print(f"Wrote plot to {args.output_dir / args.plot_name}")


if __name__ == "__main__":
    main()

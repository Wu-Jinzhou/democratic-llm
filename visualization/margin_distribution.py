#!/usr/bin/env python3
"""
Plot distribution of top-1 margin over runner-up per question.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import apply_style, style_axes


def load_listwise(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def compute_margins(records: List[dict]) -> List[float]:
    margins = []
    for rec in records:
        rankings = rec.get("rankings", [])
        if not rankings:
            continue
        counts = Counter([r[0] for r in rankings if r])
        if not counts:
            continue
        values = sorted(counts.values(), reverse=True)
        top = values[0]
        second = values[1] if len(values) > 1 else 0
        margin = (top - second) / len(rankings)
        margins.append(margin)
    return margins


def save_csv(margins: List[float], path: Path) -> None:
    df = pd.DataFrame({"margin": margins})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot(margins: List[float], path: Path) -> None:
    apply_style(grid=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(margins, bins=20, color="#4C78A8", edgecolor="none")
    ax.set_xlabel("Top-1 margin over runner-up")
    ax.set_ylabel("Questions")
    ax.set_title("Distribution of top-1 margins")
    style_axes(ax, grid=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot top-1 margin distribution.")
    parser.add_argument(
        "--listwise",
        type=Path,
        default=Path("artifacts/evaluations/listwise.jsonl"),
        help="Listwise rankings JSONL.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("visualization/output"))
    parser.add_argument("--plot-name", default="margin_distribution.png")
    parser.add_argument("--csv-name", default="margin_distribution.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_listwise(args.listwise)
    margins = compute_margins(records)
    if not margins:
        raise RuntimeError("No margins computed.")
    save_csv(margins, args.output_dir / args.csv_name)
    plot(margins, args.output_dir / args.plot_name)
    print(f"Wrote plot to {args.output_dir / args.plot_name}")


if __name__ == "__main__":
    main()

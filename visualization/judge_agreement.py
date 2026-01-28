#!/usr/bin/env python3
"""
Plot judge agreement metrics from listwise rankings.
Produces distributions of top-1 entropy and majority margin.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import List, Tuple

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


def compute_agreement(records: List[dict]) -> Tuple[List[float], List[float]]:
    entropies = []
    margins = []
    for rec in records:
        rankings = rec.get("rankings", [])
        models = rec.get("models", [])
        if not rankings:
            continue
        counts = Counter([r[0] for r in rankings if r])
        if not counts:
            continue
        total = len(rankings)
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        if models:
            entropy /= math.log(len(models))
        values = sorted(counts.values(), reverse=True)
        top = values[0]
        second = values[1] if len(values) > 1 else 0
        margin = (top - second) / total
        entropies.append(entropy)
        margins.append(margin)
    return entropies, margins


def save_csv(entropies: List[float], margins: List[float], path: Path) -> None:
    df = pd.DataFrame({"entropy_normalized": entropies, "majority_margin": margins})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot(entropies: List[float], margins: List[float], path: Path) -> None:
    apply_style(grid=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(entropies, bins=20, color="#4C78A8", edgecolor="none")
    axes[0].set_title("Top-1 entropy")
    axes[0].set_xlabel("Normalized entropy")
    axes[0].set_ylabel("Questions")
    style_axes(axes[0], grid=False)

    axes[1].hist(margins, bins=20, color="#4C78A8", edgecolor="none")
    axes[1].set_title("Majority margin")
    axes[1].set_xlabel("Top-1 margin over runner-up")
    axes[1].set_ylabel("Questions")
    style_axes(axes[1], grid=False)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot judge agreement metrics.")
    parser.add_argument(
        "--listwise",
        type=Path,
        default=Path("artifacts/evaluations/listwise.jsonl"),
        help="Listwise rankings JSONL.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("visualization/output"))
    parser.add_argument("--plot-name", default="judge_agreement.png")
    parser.add_argument("--csv-name", default="judge_agreement.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_listwise(args.listwise)
    entropies, margins = compute_agreement(records)
    if not entropies:
        raise RuntimeError("No agreement metrics computed.")
    save_csv(entropies, margins, args.output_dir / args.csv_name)
    plot(entropies, margins, args.output_dir / args.plot_name)
    print(f"Wrote plot to {args.output_dir / args.plot_name}")


if __name__ == "__main__":
    main()

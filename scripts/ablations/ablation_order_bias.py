#!/usr/bin/env python3
"""
Check for order bias in listwise judging.

Requires listwise JSONL with either:
- judge_orders (list of presented model order per judge), or
- judge_label_to_model (label -> model map per judge).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_listwise(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_orders(rec: dict) -> List[List[str]]:
    if "judge_orders" in rec and rec["judge_orders"]:
        return rec["judge_orders"]
    if "judge_label_to_model" in rec and rec["judge_label_to_model"]:
        orders = []
        for mapping in rec["judge_label_to_model"]:
            if not isinstance(mapping, dict):
                continue
            labels = sorted(mapping.keys())
            orders.append([mapping[label] for label in labels])
        if orders:
            return orders
    raise KeyError(
        "Missing judge_orders or judge_label_to_model in listwise output. "
        "Rerun evaluation with updated scripts."
    )


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description="Order bias diagnostic for listwise judging.")
    parser.add_argument(
        "--listwise",
        type=Path,
        default=Path("artifacts/evaluations/listwise.jsonl"),
        help="Listwise JSONL output from evaluate_constitution.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluations/order_bias.json"),
        help="Output JSON summary.",
    )
    args = parser.parse_args()

    records = load_listwise(args.listwise)
    if not records:
        raise RuntimeError("No listwise records found.")

    presented_positions = []
    final_ranks = []
    top1_counts = None
    rank_sum = None
    n_models = None

    total_rankings = 0
    for rec in records:
        rankings = rec.get("rankings") or []
        if not rankings:
            continue
        orders = extract_orders(rec)
        if len(orders) != len(rankings):
            raise RuntimeError("Judge order count does not match ranking count.")
        for ranking, order in zip(rankings, orders):
            if n_models is None:
                n_models = len(order)
                top1_counts = np.zeros(n_models, dtype=float)
                rank_sum = np.zeros(n_models, dtype=float)
            if len(order) != n_models or len(ranking) != n_models:
                continue
            pos_presented = {m: i + 1 for i, m in enumerate(order)}
            pos_ranked = {m: i + 1 for i, m in enumerate(ranking)}
            for m in order:
                presented_positions.append(pos_presented[m])
                final_ranks.append(pos_ranked[m])
            # top-1 rate by presented position
            winner = ranking[0]
            top1_counts[pos_presented[winner] - 1] += 1
            rank_sum += np.array([pos_ranked[m] for m in order], dtype=float)
            total_rankings += 1

    if n_models is None or top1_counts is None or rank_sum is None:
        raise RuntimeError("No valid rankings with orders found.")

    x = np.array(presented_positions, dtype=float)
    y = np.array(final_ranks, dtype=float)
    corr = pearson_corr(x, y)
    mean_rank_by_position = (rank_sum / max(total_rankings, 1)).tolist()
    top1_rate_by_position = (top1_counts / max(total_rankings, 1)).tolist()

    output = {
        "num_rankings": total_rankings,
        "num_models": n_models,
        "pearson_corr_presented_vs_rank": corr,
        "mean_rank_by_presented_position": mean_rank_by_position,
        "top1_rate_by_presented_position": top1_rate_by_position,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"Wrote order bias summary to {args.output}")
    print(f"Pearson corr (presented pos vs rank): {corr:.6f}")


if __name__ == "__main__":
    main()

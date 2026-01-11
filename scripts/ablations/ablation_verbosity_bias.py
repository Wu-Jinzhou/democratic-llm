#!/usr/bin/env python3
"""
Check whether response length correlates with preference in listwise judging.

Uses listwise JSONL containing responses and rankings.
Reports correlation between length and rank, and how often longer responses win pairwise.
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


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denom)


def response_lengths(text: str) -> Tuple[int, int]:
    chars = len(text)
    words = len(text.split())
    return chars, words


def main() -> None:
    parser = argparse.ArgumentParser(description="Verbosity bias diagnostic for listwise judging.")
    parser.add_argument(
        "--listwise",
        type=Path,
        default=Path("artifacts/evaluations/listwise.jsonl"),
        help="Listwise JSONL output from evaluate_constitution.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluations/verbosity_bias.json"),
        help="Output JSON summary.",
    )
    args = parser.parse_args()

    records = load_listwise(args.listwise)
    if not records:
        raise RuntimeError("No listwise records found.")

    lengths_chars = []
    lengths_words = []
    ranks = []
    pair_longer_win = 0
    pair_total = 0

    for rec in records:
        responses = rec.get("responses") or {}
        rankings = rec.get("rankings") or []
        if not responses or not rankings:
            continue
        # precompute lengths
        lengths = {}
        for m, text in responses.items():
            if not isinstance(text, str):
                continue
            lengths[m] = response_lengths(text)
        for ranking in rankings:
            if not ranking:
                continue
            pos = {m: i + 1 for i, m in enumerate(ranking)}
            for m in ranking:
                if m not in lengths:
                    continue
                c, w = lengths[m]
                lengths_chars.append(c)
                lengths_words.append(w)
                ranks.append(pos[m])

            # pairwise: does longer response win?
            for i in range(len(ranking)):
                for j in range(i + 1, len(ranking)):
                    mi = ranking[i]
                    mj = ranking[j]
                    if mi not in lengths or mj not in lengths:
                        continue
                    ci, wi = lengths[mi]
                    cj, wj = lengths[mj]
                    # winner is mi (higher rank)
                    if ci == cj:
                        continue
                    pair_total += 1
                    if ci > cj:
                        pair_longer_win += 1

    x_chars = np.array(lengths_chars, dtype=float)
    x_words = np.array(lengths_words, dtype=float)
    y_rank = np.array(ranks, dtype=float)
    corr_chars = pearson_corr(x_chars, y_rank)
    corr_words = pearson_corr(x_words, y_rank)

    longer_win_rate = pair_longer_win / pair_total if pair_total > 0 else float("nan")

    output = {
        "num_rankings": len(ranks),
        "pearson_corr_chars_vs_rank": corr_chars,
        "pearson_corr_words_vs_rank": corr_words,
        "pairwise_longer_win_rate": longer_win_rate,
        "pairwise_total": pair_total,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"Wrote verbosity bias summary to {args.output}")
    print(f"Pearson corr (chars vs rank): {corr_chars:.6f}")
    print(f"Pearson corr (words vs rank): {corr_words:.6f}")
    print(f"Pairwise longer-win rate: {longer_win_rate:.4f} ({pair_total} pairs)")


if __name__ == "__main__":
    main()

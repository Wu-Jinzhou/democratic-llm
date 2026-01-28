#!/usr/bin/env python3
"""
Judge reliability diagnostics for listwise rankings.

Given `artifacts/evaluations/listwise.jsonl`, this script computes:
- Per-question mean pairwise Kendall tau between the J judge rankings
- Top-1 agreement rate (fraction of judges choosing the same winner)
- Fleiss' kappa for the top-1 choice across all questions

Outputs:
- JSON summary (overall)
- Optional CSV (per-question metrics)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def kendall_tau(a: List[str], b: List[str]) -> float:
    """Kendall tau for two full rankings over the same items."""
    if len(a) != len(b):
        raise ValueError("Rankings must have the same length.")
    m = len(a)
    if m < 2:
        return 1.0
    pos_a = {x: i for i, x in enumerate(a)}
    pos_b = {x: i for i, x in enumerate(b)}
    items = list(pos_a.keys())
    # Ensure both are permutations over same set
    if set(pos_a.keys()) != set(pos_b.keys()):
        raise ValueError("Rankings must contain the same items.")
    discordant = 0
    for i in range(m):
        for j in range(i + 1, m):
            xi, xj = items[i], items[j]
            if (pos_a[xi] - pos_a[xj]) * (pos_b[xi] - pos_b[xj]) < 0:
                discordant += 1
    total_pairs = m * (m - 1) // 2
    # tau = 1 - 2 * D / total_pairs
    return 1.0 - (2.0 * discordant / float(total_pairs))


def mean_pairwise_kendall(ranks: List[List[str]]) -> Tuple[float, float, float]:
    """Return (mean, min, max) Kendall tau over all judge pairs for one question."""
    n = len(ranks)
    if n < 2:
        return 1.0, 1.0, 1.0
    taus: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            taus.append(kendall_tau(ranks[i], ranks[j]))
    return float(sum(taus) / len(taus)), float(min(taus)), float(max(taus))


def fleiss_kappa_top1(per_item_top1_counts: List[Dict[str, int]], n_raters: int) -> float:
    """
    Fleiss' kappa for categorical judgments, where each item has counts per category.
    Here categories are models and the judgment is the top-1 choice.
    """
    if not per_item_top1_counts:
        return float("nan")
    if n_raters < 2:
        return float("nan")

    # Global category proportions p_j
    cat_totals: Dict[str, int] = {}
    for counts in per_item_top1_counts:
        for cat, c in counts.items():
            cat_totals[cat] = cat_totals.get(cat, 0) + int(c)

    N = len(per_item_top1_counts)
    denom = N * n_raters
    p: Dict[str, float] = {cat: total / denom for cat, total in cat_totals.items()}
    P_e = sum(v * v for v in p.values())

    # Per-item agreement P_i
    P_i_vals: List[float] = []
    for counts in per_item_top1_counts:
        s = 0
        for c in counts.values():
            s += c * (c - 1)
        P_i_vals.append(s / (n_raters * (n_raters - 1)))
    P_bar = sum(P_i_vals) / N
    if math.isclose(1.0 - P_e, 0.0):
        return float("nan")
    return (P_bar - P_e) / (1.0 - P_e)


@dataclass
class PerQuestion:
    question_id: str
    clause_id: int
    n_models: int
    n_judges: int
    mean_kendall_tau: float
    min_kendall_tau: float
    max_kendall_tau: float
    top1_agreement: float
    top1_margin: float
    top1_winner: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge reliability diagnostics for listwise rankings.")
    parser.add_argument("--listwise", type=Path, default=Path("artifacts/evaluations/listwise.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-question-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None  # type: ignore

    records = list(iter_jsonl(args.listwise))
    it = tqdm(records, desc="Computing judge reliability") if tqdm else records

    per_question: List[PerQuestion] = []
    per_item_top1_counts: List[Dict[str, int]] = []

    for obj in it:
        question_id = str(obj.get("question_id"))
        clause_id = int(obj.get("clause_id"))
        rankings = obj.get("rankings") or []
        if not rankings:
            continue
        # Ensure list[list[str]]
        ranks: List[List[str]] = [[str(x) for x in r] for r in rankings]
        n_judges = len(ranks)
        n_models = len(ranks[0]) if ranks else 0
        mean_tau, min_tau, max_tau = mean_pairwise_kendall(ranks)

        top1 = [r[0] for r in ranks if r]
        c = Counter(top1)
        if not c:
            continue
        top1_winner, top1_wins = c.most_common(1)[0]
        sorted_counts = sorted(c.values(), reverse=True)
        second = sorted_counts[1] if len(sorted_counts) > 1 else 0
        top1_agreement = top1_wins / n_judges
        top1_margin = (top1_wins - second) / n_judges

        per_item_top1_counts.append(dict(c))
        per_question.append(
            PerQuestion(
                question_id=question_id,
                clause_id=clause_id,
                n_models=n_models,
                n_judges=n_judges,
                mean_kendall_tau=mean_tau,
                min_kendall_tau=min_tau,
                max_kendall_tau=max_tau,
                top1_agreement=top1_agreement,
                top1_margin=top1_margin,
                top1_winner=top1_winner,
            )
        )

    if not per_question:
        raise RuntimeError(f"No usable rankings found in {args.listwise}")

    # Overall summaries
    taus = [p.mean_kendall_tau for p in per_question]
    agreements = [p.top1_agreement for p in per_question]
    margins = [p.top1_margin for p in per_question]
    n_judges = per_question[0].n_judges

    summary = {
        "method": "judge-reliability",
        "listwise_path": str(args.listwise),
        "n_questions": len(per_question),
        "n_models": per_question[0].n_models,
        "n_judges": n_judges,
        "kendall_tau_mean": float(sum(taus) / len(taus)),
        "kendall_tau_median": float(sorted(taus)[len(taus) // 2]),
        "top1_agreement_mean": float(sum(agreements) / len(agreements)),
        "top1_margin_mean": float(sum(margins) / len(margins)),
        "fleiss_kappa_top1": float(fleiss_kappa_top1(per_item_top1_counts, n_raters=n_judges)),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote judge reliability summary to {args.output}")

    if args.per_question_csv is not None:
        args.per_question_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.per_question_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "question_id",
                    "clause_id",
                    "n_models",
                    "n_judges",
                    "mean_kendall_tau",
                    "min_kendall_tau",
                    "max_kendall_tau",
                    "top1_winner",
                    "top1_agreement",
                    "top1_margin",
                ]
            )
            for p in per_question:
                w.writerow(
                    [
                        p.question_id,
                        p.clause_id,
                        p.n_models,
                        p.n_judges,
                        f"{p.mean_kendall_tau:.6f}",
                        f"{p.min_kendall_tau:.6f}",
                        f"{p.max_kendall_tau:.6f}",
                        p.top1_winner,
                        f"{p.top1_agreement:.6f}",
                        f"{p.top1_margin:.6f}",
                    ]
                )
        print(f"Wrote per-question reliability CSV to {args.per_question_csv}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Clause-level comparison: Soft Panel vs Full PRISM.

Uses `preferences.jsonl` to compute, per clause:
- vote-level win rate (wins / total votes)
- majority win rate (fraction of questions where the pairwise majority favors Soft)
- mean margin (mean over questions of (wins_soft - wins_full) / num_judges)

Outputs a CSV sorted by the win rate.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_clause_text_map(listwise_path: Path) -> Dict[int, str]:
    if not listwise_path.exists():
        return {}
    out: Dict[int, str] = {}
    for obj in iter_jsonl(listwise_path):
        try:
            cid = int(obj.get("clause_id"))
        except Exception:
            continue
        if cid not in out:
            clause = obj.get("clause")
            if isinstance(clause, str):
                out[cid] = clause
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-clause Soft Panel vs Full PRISM diagnostics.")
    parser.add_argument("--preferences", type=Path, default=Path("artifacts/evaluations/preferences.jsonl"))
    parser.add_argument(
        "--soft-model",
        default="checkpoints/llama3.1-8b-soft-panel",
        help="Model id used for Soft Panel.",
    )
    parser.add_argument(
        "--full-model",
        default="checkpoints/llama3.1-8b-full-prism",
        help="Model id used for Full PRISM.",
    )
    parser.add_argument(
        "--listwise",
        type=Path,
        default=Path("artifacts/evaluations/listwise.jsonl"),
        help="Optional listwise file to attach clause text.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    soft = str(args.soft_model)
    full = str(args.full_model)

    clause_text = build_clause_text_map(args.listwise)

    wins_soft_by_clause: Dict[int, int] = defaultdict(int)
    wins_full_by_clause: Dict[int, int] = defaultdict(int)
    n_questions_by_clause: Dict[int, int] = defaultdict(int)
    maj_soft_by_clause: Dict[int, int] = defaultdict(int)
    margin_sum_by_clause: Dict[int, float] = defaultdict(float)
    n_judges_by_clause: Dict[int, int] = defaultdict(int)

    for obj in iter_jsonl(args.preferences):
        mi = str(obj.get("model_i"))
        mj = str(obj.get("model_j"))
        if {mi, mj} != {soft, full}:
            continue
        clause_id = int(obj.get("clause_id"))
        n_judges = int(obj.get("num_judges", 1))
        wins_i = int(obj.get("wins_i", 0))
        wins_j = int(obj.get("wins_j", 0))
        majority = obj.get("majority_winner")

        wins_soft = wins_i if mi == soft else wins_j
        wins_full = wins_j if mi == soft else wins_i

        wins_soft_by_clause[clause_id] += wins_soft
        wins_full_by_clause[clause_id] += wins_full
        n_questions_by_clause[clause_id] += 1
        maj_soft_by_clause[clause_id] += 1 if majority == soft else 0
        margin_sum_by_clause[clause_id] += (wins_soft - wins_full) / float(n_judges)
        n_judges_by_clause[clause_id] = n_judges

    rows = []
    for clause_id, n_q in n_questions_by_clause.items():
        n_judges = n_judges_by_clause.get(clause_id, 1)
        total_votes = n_q * n_judges
        win_rate_votes = wins_soft_by_clause[clause_id] / total_votes if total_votes else 0.0
        win_rate_majority = maj_soft_by_clause[clause_id] / n_q if n_q else 0.0
        mean_margin = margin_sum_by_clause[clause_id] / n_q if n_q else 0.0
        rows.append(
            {
                "clause_id": clause_id,
                "n_questions": n_q,
                "n_judges": n_judges,
                "win_rate_votes_soft": win_rate_votes,
                "win_rate_majority_soft": win_rate_majority,
                "mean_margin_soft": mean_margin,
                "clause": clause_text.get(clause_id, ""),
            }
        )

    # Sort descending: strongest soft-vs-full clauses first
    rows.sort(key=lambda r: (-(r["win_rate_votes_soft"]), -(r["win_rate_majority_soft"]), r["clause_id"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "clause_id",
                "n_questions",
                "n_judges",
                "win_rate_votes_soft",
                "win_rate_majority_soft",
                "mean_margin_soft",
                "clause",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote clause-level soft-vs-full report to {args.output}")


if __name__ == "__main__":
    main()


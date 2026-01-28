#!/usr/bin/env python3
"""
Pairwise effect sizes from `preferences.jsonl`.

Computes win-rates (vote-level and majority-level) for each model pair, with optional
question-level bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def canonical_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


@dataclass
class PairRow:
    model_a: str
    model_b: str
    n_questions: int
    n_judges: int
    win_rate_a_votes: float
    win_rate_a_majority: float
    mean_margin_a: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None


def bootstrap_ci(
    wins_a: np.ndarray,
    n_judges: int,
    samples: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = wins_a.size
    if n == 0:
        return float("nan"), float("nan")
    # (samples, n) indices; for n=3000 and samples=1000 this is ~3M ints (~24MB)
    idx = rng.integers(0, n, size=(samples, n), dtype=np.int32)
    boot_rates = wins_a[idx].sum(axis=1) / (float(n_judges) * float(n))
    lo, hi = np.quantile(boot_rates, [0.025, 0.975])
    return float(lo), float(hi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise effect sizes from preferences.jsonl")
    parser.add_argument("--preferences", type=Path, default=Path("artifacts/evaluations/preferences.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None  # type: ignore

    # Accumulate per-pair per-question wins for bootstrap + summary
    per_pair_wins: Dict[Tuple[str, str], List[int]] = {}
    per_pair_majority_a: Dict[Tuple[str, str], List[int]] = {}
    per_pair_n_judges: Dict[Tuple[str, str], int] = {}

    items = list(iter_jsonl(args.preferences))
    it = tqdm(items, desc="Loading preferences") if tqdm else items
    for obj in it:
        mi = str(obj["model_i"])
        mj = str(obj["model_j"])
        a, b = canonical_pair(mi, mj)
        n_judges = int(obj.get("num_judges", 1))
        wins_i = int(obj.get("wins_i", 0))
        wins_j = int(obj.get("wins_j", 0))
        maj = obj.get("majority_winner")

        if (a, b) not in per_pair_wins:
            per_pair_wins[(a, b)] = []
            per_pair_majority_a[(a, b)] = []
        per_pair_n_judges[(a, b)] = n_judges

        wins_a = wins_i if mi == a else wins_j
        per_pair_wins[(a, b)].append(wins_a)
        per_pair_majority_a[(a, b)].append(1 if maj == a else 0)

    rows: List[PairRow] = []
    for (a, b), wins_list in per_pair_wins.items():
        wins_a = np.asarray(wins_list, dtype=np.int16)
        n_questions = int(wins_a.size)
        n_judges = int(per_pair_n_judges[(a, b)])
        total_votes = n_questions * n_judges
        win_rate_votes = float(wins_a.sum() / float(total_votes)) if total_votes > 0 else float("nan")
        maj_a = np.asarray(per_pair_majority_a[(a, b)], dtype=np.int8)
        win_rate_majority = float(maj_a.mean()) if n_questions > 0 else float("nan")

        # Mean per-question margin in [-1, 1] where 1 means unanimous for A
        mean_margin = float(((wins_a - (n_judges - wins_a)) / n_judges).mean())

        lo = hi = None
        if args.bootstrap_samples and args.bootstrap_samples > 0:
            lo, hi = bootstrap_ci(wins_a, n_judges=n_judges, samples=args.bootstrap_samples, seed=args.seed)

        rows.append(
            PairRow(
                model_a=a,
                model_b=b,
                n_questions=n_questions,
                n_judges=n_judges,
                win_rate_a_votes=win_rate_votes,
                win_rate_a_majority=win_rate_majority,
                mean_margin_a=mean_margin,
                ci_lower=lo,
                ci_upper=hi,
            )
        )

    # Sort: strongest vote-level winners first
    rows.sort(key=lambda r: (-r.win_rate_a_votes, r.model_a, r.model_b))

    out: Dict[str, Any] = {
        "method": "pairwise-effects",
        "preferences_path": str(args.preferences),
        "bootstrap_samples": int(args.bootstrap_samples),
        "seed": int(args.seed),
        "pairs": [
            {
                "model_a": r.model_a,
                "model_b": r.model_b,
                "n_questions": r.n_questions,
                "n_judges": r.n_judges,
                "win_rate_a_votes": r.win_rate_a_votes,
                "win_rate_a_majority": r.win_rate_a_majority,
                "mean_margin_a": r.mean_margin_a,
                "ci_lower": r.ci_lower,
                "ci_upper": r.ci_upper,
            }
            for r in rows
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".csv":
        df = pd.DataFrame(out["pairs"])
        df.to_csv(args.output, index=False)
    else:
        args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote pairwise effects to {args.output}")


if __name__ == "__main__":
    main()


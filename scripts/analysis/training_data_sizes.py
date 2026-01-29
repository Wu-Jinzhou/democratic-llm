#!/usr/bin/env python3
"""
Compute simple dataset-size statistics for prepared DPO JSONL files.

Outputs per-dataset:
- pairs: number of JSONL records
- raters: unique user_id count
- interactions: unique interaction_id count
- weight stats (if present)

This is intended for keeping Appendix tables up-to-date.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno}: expected object, got {type(obj)}")
            yield obj


def _weight_stats(weights: List[float]) -> Optional[Dict[str, float]]:
    if not weights:
        return None
    return {
        "min": float(min(weights)),
        "max": float(max(weights)),
        "mean": float(statistics.fmean(weights)),
        "median": float(statistics.median(weights)),
    }


def compute_stats(path: Path) -> Dict[str, Any]:
    pairs = 0
    raters: set[str] = set()
    interactions: set[str] = set()
    weights: List[float] = []
    pairs_per_rater: Counter[str] = Counter()

    for row in iter_jsonl(path):
        pairs += 1
        user_id = row.get("user_id")
        if isinstance(user_id, str) and user_id:
            raters.add(user_id)
            pairs_per_rater[user_id] += 1
        interaction_id = row.get("interaction_id")
        if isinstance(interaction_id, str) and interaction_id:
            interactions.add(interaction_id)
        w = row.get("weight")
        if isinstance(w, (int, float)):
            weights.append(float(w))

    per_rater_counts = list(pairs_per_rater.values())
    per_rater_stats = None
    if per_rater_counts:
        per_rater_stats = {
            "min": int(min(per_rater_counts)),
            "max": int(max(per_rater_counts)),
            "mean": float(statistics.fmean(per_rater_counts)),
            "median": float(statistics.median(per_rater_counts)),
        }

    return {
        "path": str(path),
        "pairs": pairs,
        "raters": len(raters),
        "interactions": len(interactions),
        "weight": _weight_stats(weights),
        "pairs_per_rater": per_rater_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute dataset-size stats for prepared DPO JSONL files.")
    parser.add_argument(
        "--files",
        nargs="*",
        type=Path,
        default=[],
        help="Explicit JSONL files to analyze (overrides --data-dir defaults if provided).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("artifacts/data"),
        help="Directory containing prepared datasets (default: artifacts/data).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON results. If omitted, prints to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.files:
        files = args.files
    else:
        files = [
            args.data_dir / "full.jsonl",
            args.data_dir / "soft_panel.jsonl",
            args.data_dir / "hard_panel.jsonl",
            args.data_dir / "us_rep.jsonl",
        ]

    results: Dict[str, Any] = {"datasets": []}
    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        results["datasets"].append(compute_stats(path))

    if args.output is None:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {len(results['datasets'])} dataset summaries to {args.output}")


if __name__ == "__main__":
    main()


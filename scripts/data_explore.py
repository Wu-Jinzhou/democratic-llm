#!/usr/bin/env python3
"""
Lightweight explorers for PRISM survey/utterance schemas.

Examples:
  python scripts/data_explore.py --survey prism-alignment/survey.jsonl --columns study_locale,gender,education --limit 2000
  python scripts/data_explore.py --utterances prism-alignment/utterances.jsonl --columns conversation_type,model_name
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def show_unique(df: pd.DataFrame, columns: Iterable[str], top_n: int = 20) -> None:
    for col in columns:
        if col not in df.columns:
            print(f"[warn] column {col} not found in dataframe")
            continue
        value_counts = df[col].value_counts(dropna=False).head(top_n)
        print(f"\n=== {col} (top {top_n}) ===")
        print(value_counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore PRISM survey/utterance columns.")
    parser.add_argument("--survey", type=Path, default=None, help="Path to survey.jsonl")
    parser.add_argument("--utterances", type=Path, default=None, help="Path to utterances.jsonl")
    parser.add_argument(
        "--columns",
        type=lambda s: [c.strip() for c in s.split(",") if c.strip()],
        default=[],
        help="Comma-separated column names to summarize.",
    )
    parser.add_argument("--limit", type=int, default=5000, help="Number of rows to sample for uniqueness stats.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.columns:
        raise SystemExit("Provide --columns to inspect.")
    if args.survey:
        df = pd.read_json(args.survey, lines=True, nrows=args.limit)
        print(f"[survey] loaded {len(df)} rows from {args.survey}")
        show_unique(df, args.columns)
    if args.utterances:
        df = pd.read_json(args.utterances, lines=True, nrows=args.limit)
        print(f"[utterances] loaded {len(df)} rows from {args.utterances}")
        show_unique(df, args.columns)


if __name__ == "__main__":
    main()

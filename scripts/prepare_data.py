#!/usr/bin/env python3
"""
Build DPO-ready datasets from PRISM data using hard/soft sortition or full US-REP subset.

Outputs a JSONL with columns:
- prompt: user prompt text
- chosen: preferred model response
- rejected: alternative model response
- user_id: rater id
- interaction_id: PRISM interaction id
- weight: (optional) sample weight for soft panel training
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from tqdm import tqdm

# Ensure local imports work even when executed as a script
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from sortition import (  # type: ignore
    PanelConfig,
    estimate_selection_probabilities,
    load_panel_config,
    prepare_panel_data,
    sample_panel,
)


def load_jsonl(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_json(path, lines=True, nrows=nrows)


def build_pairs(utterances: pd.DataFrame) -> List[dict]:
    """Construct (prompt, chosen, rejected) pairs per interaction."""
    pairs: List[dict] = []
    grouped = utterances.groupby("interaction_id")
    for interaction_id, group in tqdm(grouped, desc="Building pairs"):
        chosen_rows = group[group["if_chosen"] == True].sort_values("score", ascending=False)
        rejected_rows = group[group["if_chosen"] == False].sort_values("score", ascending=False)
        if chosen_rows.empty or rejected_rows.empty:
            continue
        best = chosen_rows.iloc[0]
        prompt = best["user_prompt"]
        chosen_resp = best["model_response"]
        user_id = best["user_id"]
        for _, rej in rejected_rows.iterrows():
            pairs.append(
                {
                    "prompt": prompt,
                    "chosen": chosen_resp,
                    "rejected": rej["model_response"],
                    "user_id": user_id,
                    "interaction_id": interaction_id,
                }
            )
    return pairs


def attach_weights(pairs: List[dict], weights: pd.Series | None) -> List[dict]:
    if weights is None:
        for p in pairs:
            p["weight"] = 1.0
        return pairs
    weight_map = weights.to_dict()
    for p in pairs:
        p["weight"] = float(weight_map.get(p["user_id"], 1.0))
    return pairs


def save_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def prepare_hard_panel(
    survey_df: pd.DataFrame,
    utterances_df: pd.DataFrame,
    panel_config: PanelConfig,
    panel_seed: int,
    panel_algorithm: str,
) -> List[dict]:
    prepared = prepare_panel_data(survey_df, panel_config)
    panel = sample_panel(
        prepared,
        attrs=panel_config.attributes,
        panel_size=panel_config.panel_size,
        algorithm=panel_algorithm,
        rng=random.Random(panel_seed),
    )
    panel_ids = set(panel["user_id"].tolist())
    filtered = utterances_df[utterances_df["user_id"].isin(panel_ids)]
    return attach_weights(build_pairs(filtered), None)


def prepare_soft_panel(
    survey_df: pd.DataFrame,
    utterances_df: pd.DataFrame,
    panel_config: PanelConfig,
    num_samples: int,
    seed: int,
    num_workers: int,
    panel_algorithm: str,
) -> List[dict]:
    prepared = prepare_panel_data(survey_df, panel_config)
    probabilities = estimate_selection_probabilities(
        prepared,
        attrs=panel_config.attributes,
        panel_size=panel_config.panel_size,
        num_samples=num_samples,
        rng_seed=seed,
        num_workers=num_workers,
        algorithm=panel_algorithm,
    )
    weights = probabilities.copy()
    weights.index = prepared["user_id"].values
    filtered = utterances_df[utterances_df["user_id"].isin(prepared["user_id"])]
    pairs = build_pairs(filtered)
    return attach_weights(pairs, weights=weights)


def prepare_us_rep(
    survey_df: pd.DataFrame,
    utterances_df: pd.DataFrame,
) -> List[dict]:
    ids = survey_df.loc[survey_df["included_in_US_REP"] == True, "user_id"]
    filtered = utterances_df[utterances_df["user_id"].isin(ids)]
    return attach_weights(build_pairs(filtered), None)


def prepare_full(
    survey_df: pd.DataFrame,
    utterances_df: pd.DataFrame,
) -> List[dict]:
    """Use all raters / utterances without filtering or weighting."""
    return attach_weights(build_pairs(utterances_df), None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DPO datasets from PRISM.")
    parser.add_argument("--survey", type=Path, default=Path("prism-alignment/survey.jsonl"))
    parser.add_argument("--utterances", type=Path, default=Path("prism-alignment/utterances.jsonl"))
    parser.add_argument(
        "--mode",
        choices=["hard", "soft", "us_rep", "full"],
        required=True,
        help="Which dataset variant to produce.",
    )
    parser.add_argument("--panel-config", type=Path, default=Path("configs/panel_config.yaml"))
    parser.add_argument(
        "--panel-algorithm",
        choices=["legacy", "leximin", "random"],
        default="legacy",
        help="Panel selection algorithm (Sortition Foundation LEGACY/LEXIMIN).",
    )
    parser.add_argument("--panel-seed", type=int, default=0)
    parser.add_argument("--num-panel-samples", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel workers for soft panel sampling.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    survey_df = load_jsonl(args.survey)
    utterances_df = load_jsonl(args.utterances)

    if args.mode in {"hard", "soft"}:
        panel_config = load_panel_config(args.panel_config)
    else:
        panel_config = None  # type: ignore

    if args.mode == "hard":
        records = prepare_hard_panel(
            survey_df=survey_df,
            utterances_df=utterances_df,
            panel_config=panel_config,
            panel_seed=args.panel_seed,
            panel_algorithm=args.panel_algorithm,
        )
    elif args.mode == "soft":
        records = prepare_soft_panel(
            survey_df=survey_df,
            utterances_df=utterances_df,
            panel_config=panel_config,
            num_samples=args.num_panel_samples,
            seed=args.panel_seed,
            num_workers=args.num_workers,
            panel_algorithm=args.panel_algorithm,
        )
    else:
        records = prepare_us_rep(survey_df=survey_df, utterances_df=utterances_df)
    if args.mode == "full":
        records = prepare_full(survey_df=survey_df, utterances_df=utterances_df)

    save_jsonl(records, args.output)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()

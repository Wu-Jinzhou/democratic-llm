#!/usr/bin/env python3
"""
Build DPO-ready datasets from PRISM data using hard/soft sortition or full US-REP subset.

Outputs a JSONL with columns:
- prompt: list of {"role","content"} messages (conversational format) or raw text
- chosen: list of assistant messages or raw text
- rejected: list of assistant messages or raw text
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
from typing import Iterable, List, Optional

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
    if path.exists():
        with path.open("rb") as f:
            header = f.read(200)
        if b"git-lfs" in header and b"version https://git-lfs.github.com/spec/v1" in header:
            raise RuntimeError(
                f"{path} looks like a Git LFS pointer file. "
                "Run `git lfs pull` in the dataset repo (prism-alignment) to fetch the real JSONL."
            )
    try:
        return pd.read_json(path, lines=True, nrows=nrows)
    except ValueError:
        records: List[dict] = []
        bad_lines = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if nrows is not None and len(records) >= nrows:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    bad_lines += 1
        if bad_lines:
            print(f"Warning: skipped {bad_lines} malformed lines in {path}")
        return pd.DataFrame.from_records(records)


def _normalize_column(df: pd.DataFrame, target: str, candidates: List[str]) -> pd.DataFrame:
    if target in df.columns:
        return df
    lower = {col.lower(): col for col in df.columns}
    for cand in [target] + candidates:
        found = lower.get(cand.lower())
        if found:
            df = df.rename(columns={found: target})
            print(f"Mapped column '{found}' -> '{target}'")
            return df
    for col in df.columns:
        series = df[col]
        if series.dtype != object:
            continue
        sample = series.dropna().head(1)
        if sample.empty:
            continue
        value = sample.iloc[0]
        if isinstance(value, dict):
            for cand in [target] + candidates:
                if cand in value:
                    df[target] = series.apply(
                        lambda x: x.get(cand) if isinstance(x, dict) else None
                    )
                    print(f"Extracted '{target}' from dict column '{col}'")
                    return df
    return df


def normalize_utterances(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_column(
        df,
        "user_id",
        ["rater_id", "worker_id", "participant_id", "annotator_id", "respondent_id", "uid", "user"],
    )
    df = _normalize_column(
        df,
        "interaction_id",
        ["interaction", "interactionid", "conversation_id", "conversationid"],
    )
    df = _normalize_column(
        df,
        "user_prompt",
        ["prompt", "question", "input", "user_query", "context"],
    )
    df = _normalize_column(
        df,
        "model_response",
        ["response", "completion", "output", "assistant_response", "assistant_reply"],
    )
    df = _normalize_column(
        df,
        "if_chosen",
        ["chosen", "is_chosen", "preferred", "winner"],
    )
    df = _normalize_column(
        df,
        "score",
        ["rating", "preference_score", "reward", "rank_score"],
    )
    required = ["user_id", "interaction_id", "user_prompt", "model_response", "if_chosen", "score"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(
            f"Utterances missing required columns: {missing}. Available columns: {list(df.columns)}"
        )
    return df


def normalize_survey(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_column(
        df,
        "user_id",
        ["rater_id", "worker_id", "participant_id", "annotator_id", "respondent_id", "uid", "user"],
    )
    if "user_id" not in df.columns:
        raise KeyError(
            f"Survey missing required column 'user_id'. Available columns: {list(df.columns)}"
        )
    return df


def format_pair(
    prompt: str,
    chosen: str,
    rejected: str,
    system_prompt: Optional[str],
    dataset_format: str,
) -> tuple[object, object, object]:
    if dataset_format == "raw":
        return prompt, chosen, rejected
    messages: List[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    chosen_messages = [{"role": "assistant", "content": chosen}]
    rejected_messages = [{"role": "assistant", "content": rejected}]
    return messages, chosen_messages, rejected_messages


def build_pairs(
    utterances: pd.DataFrame,
    system_prompt: Optional[str],
    dataset_format: str,
) -> List[dict]:
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
            prompt_value, chosen_value, rejected_value = format_pair(
                prompt=prompt,
                chosen=chosen_resp,
                rejected=rej["model_response"],
                system_prompt=system_prompt,
                dataset_format=dataset_format,
            )
            pairs.append(
                {
                    "prompt": prompt_value,
                    "chosen": chosen_value,
                    "rejected": rejected_value,
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
    system_prompt: Optional[str],
    dataset_format: str,
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
    return attach_weights(build_pairs(filtered, system_prompt, dataset_format), None)


def prepare_soft_panel(
    survey_df: pd.DataFrame,
    utterances_df: pd.DataFrame,
    panel_config: PanelConfig,
    num_samples: int,
    seed: int,
    num_workers: int,
    panel_algorithm: str,
    system_prompt: Optional[str],
    dataset_format: str,
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
    pairs = build_pairs(filtered, system_prompt, dataset_format)
    return attach_weights(pairs, weights=weights)


def prepare_us_rep(
    survey_df: pd.DataFrame,
    utterances_df: pd.DataFrame,
    system_prompt: Optional[str],
    dataset_format: str,
) -> List[dict]:
    ids = survey_df.loc[survey_df["included_in_US_REP"] == True, "user_id"]
    filtered = utterances_df[utterances_df["user_id"].isin(ids)]
    return attach_weights(build_pairs(filtered, system_prompt, dataset_format), None)


def prepare_full(
    survey_df: pd.DataFrame,
    utterances_df: pd.DataFrame,
    system_prompt: Optional[str],
    dataset_format: str,
) -> List[dict]:
    """Use all raters / utterances without filtering or weighting."""
    return attach_weights(build_pairs(utterances_df, system_prompt, dataset_format), None)


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
    parser.add_argument("--panel-seed", type=int, default=42)
    parser.add_argument("--num-panel-samples", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel workers for soft panel sampling.")
    parser.add_argument(
        "--dataset-format",
        choices=["chat", "raw"],
        default="chat",
        help="Output dataset format for DPO (chat uses role/content messages).",
    )
    parser.add_argument("--system-prompt", default=None, help="Optional system prompt for chat format.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    survey_df = normalize_survey(load_jsonl(args.survey))
    utterances_df = normalize_utterances(load_jsonl(args.utterances))

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
            system_prompt=args.system_prompt,
            dataset_format=args.dataset_format,
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
            system_prompt=args.system_prompt,
            dataset_format=args.dataset_format,
        )
    else:
        records = prepare_us_rep(
            survey_df=survey_df,
            utterances_df=utterances_df,
            system_prompt=args.system_prompt,
            dataset_format=args.dataset_format,
        )
    if args.mode == "full":
        records = prepare_full(
            survey_df=survey_df,
            utterances_df=utterances_df,
            system_prompt=args.system_prompt,
            dataset_format=args.dataset_format,
        )

    save_jsonl(records, args.output)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()

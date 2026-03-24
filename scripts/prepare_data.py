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
    # Preserve JSON types exactly instead of relying on pandas' JSON inference, which can
    # collapse large numeric-looking identifiers into lossy numeric dtypes.
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


def _clean_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
    else:
        text = str(value).strip()
    if not text:
        return None
    if text.upper() == "EMPTY STRING":
        return None
    return text


def normalize_utterances(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_column(
        df,
        "conversation_id",
        ["conversation", "conversationid"],
    )
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
    # Normalize identifiers to strings for consistent joins
    df["user_id"] = df["user_id"].astype(str)
    if "conversation_id" in df.columns:
        df["conversation_id"] = df["conversation_id"].astype(str)
    # Drop placeholder/empty prompts and responses
    for col in ["user_prompt", "model_response"]:
        cleaned = df[col].apply(_clean_text)
        dropped = int(cleaned.isna().sum())
        if dropped:
            print(f"Warning: dropped {dropped} rows with empty {col} in utterances.jsonl")
        df[col] = cleaned
    df = df.dropna(subset=["user_prompt", "model_response"])
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
    df["user_id"] = df["user_id"].astype(str)
    return df


def normalize_conversations(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_column(
        df,
        "conversation_id",
        ["conversation", "conversationid", "interaction_id", "interactionid"],
    )
    df = _normalize_column(
        df,
        "user_id",
        ["rater_id", "worker_id", "participant_id", "annotator_id", "respondent_id", "uid", "user"],
    )
    if "conversation_id" not in df.columns:
        raise KeyError(
            f"Conversations missing required column 'conversation_id'. Available columns: {list(df.columns)}"
        )
    if "conversation_history" not in df.columns:
        raise KeyError(
            f"Conversations missing required column 'conversation_history'. Available columns: {list(df.columns)}"
        )
    df["conversation_id"] = df["conversation_id"].astype(str)
    if "user_id" in df.columns:
        df["user_id"] = df["user_id"].astype(str)
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


def messages_to_raw(messages: List[dict]) -> str:
    lines: List[str] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        elif role == "tool":
            lines.append(f"Tool: {content}")
        else:
            lines.append(str(content))
    return "\n".join(lines).strip()


def _pick_chosen(candidates: List[dict]) -> Optional[str]:
    if not candidates:
        return None
    chosen = [c for c in candidates if c.get("if_chosen") is True]
    pool = chosen if chosen else [c for c in candidates if c.get("score") is not None]
    if pool:
        best = max(pool, key=lambda c: c.get("score", float("-inf")))
        return _clean_text(best.get("content") or best.get("model_response"))
    return _clean_text(candidates[0].get("content") or candidates[0].get("model_response"))


def build_pairs_from_conversations(
    conversations: pd.DataFrame,
    utterances: pd.DataFrame,
    delta: float,
    system_prompt: Optional[str],
    dataset_format: str,
) -> List[dict]:
    """Construct (prompt, chosen, rejected) pairs per conversation turn using full history."""
    pairs: List[dict] = []
    utterances_by: dict[tuple[str, int], List[dict]] = {}
    user_prompt_by: dict[tuple[str, int], str] = {}

    for row in utterances.itertuples(index=False):
        conv_id = getattr(row, "conversation_id", None)
        turn = getattr(row, "turn", None)
        if conv_id is None or turn is None:
            continue
        key = (str(conv_id), int(turn))
        score = getattr(row, "score", None)
        if score is not None:
            try:
                score = float(score)
            except Exception:
                score = None
        content = _clean_text(getattr(row, "model_response", None))
        if not content:
            continue
        candidate = {
            "content": content,
            "score": score,
            "if_chosen": bool(getattr(row, "if_chosen", False))
            if getattr(row, "if_chosen", None) is not None
            else None,
            "user_id": getattr(row, "user_id", None),
        }
        utterances_by.setdefault(key, []).append(candidate)
        prompt_val = _clean_text(getattr(row, "user_prompt", None))
        if prompt_val and key not in user_prompt_by:
            user_prompt_by[key] = prompt_val

    missing_prev_user = 0
    missing_prev_choice = 0
    missing_current_user = 0
    skipped_candidates = 0
    skipped_turns = 0

    for conv in tqdm(
        conversations.itertuples(index=False),
        total=len(conversations),
        desc="Building pairs from conversations",
    ):
        conv_id = str(getattr(conv, "conversation_id"))
        user_id = getattr(conv, "user_id", None)
        history = getattr(conv, "conversation_history", None) or []

        user_by_turn: dict[int, str] = {}
        model_by_turn: dict[int, List[dict]] = {}
        for item in history:
            if not isinstance(item, dict):
                continue
            turn = item.get("turn")
            if turn is None:
                continue
            try:
                turn = int(turn)
            except Exception:
                continue
            role = item.get("role")
            content = _clean_text(item.get("content"))
            if not content:
                continue
            if role == "user":
                user_by_turn[turn] = content
            elif role in {"model", "assistant"}:
                model_by_turn.setdefault(turn, []).append(
                    {
                        "content": content,
                        "score": item.get("score"),
                        "if_chosen": item.get("if_chosen") is True,
                    }
                )

        # Identify turns with candidates from utterances
        turns = sorted({t for (cid, t) in utterances_by.keys() if cid == conv_id})
        for turn in turns:
            # Build full history up to current turn
            messages: List[dict] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for t in range(0, turn):
                user_msg = _clean_text(user_by_turn.get(t) or user_prompt_by.get((conv_id, t)))
                if not user_msg:
                    missing_prev_user += 1
                    continue
                candidates_prev = utterances_by.get((conv_id, t), [])
                chosen_prev = _pick_chosen(candidates_prev) or _pick_chosen(model_by_turn.get(t, []))
                chosen_prev = _clean_text(chosen_prev)
                if not chosen_prev:
                    missing_prev_choice += 1
                    continue
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": chosen_prev})

            user_msg = _clean_text(user_by_turn.get(turn) or user_prompt_by.get((conv_id, turn)))
            if not user_msg:
                missing_current_user += 1
                continue
            messages.append({"role": "user", "content": user_msg})

            candidates = utterances_by.get((conv_id, turn), [])
            if len(candidates) < 2:
                skipped_candidates += 1
                continue

            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    ci = candidates[i]
                    cj = candidates[j]
                    si = ci.get("score")
                    sj = cj.get("score")
                    if si is None or sj is None:
                        continue
                    diff = si - sj
                    if diff == 0:
                        continue
                    if diff > 0 and diff >= delta:
                        chosen_resp = ci.get("content")
                        rejected_resp = cj.get("content")
                    elif diff < 0 and (-diff) >= delta:
                        chosen_resp = cj.get("content")
                        rejected_resp = ci.get("content")
                    else:
                        continue
                    if not chosen_resp or not rejected_resp:
                        continue
                    if dataset_format == "raw":
                        prompt_value = messages_to_raw(messages)
                        chosen_value = chosen_resp
                        rejected_value = rejected_resp
                    else:
                        prompt_value = messages
                        chosen_value = [{"role": "assistant", "content": chosen_resp}]
                        rejected_value = [{"role": "assistant", "content": rejected_resp}]
                    pairs.append(
                        {
                            "prompt": prompt_value,
                            "chosen": chosen_value,
                            "rejected": rejected_value,
                            "user_id": user_id or ci.get("user_id") or cj.get("user_id"),
                            "interaction_id": f"{conv_id}:{turn}",
                        }
                    )
        if not turns:
            skipped_turns += 1
    if missing_prev_user or missing_prev_choice or missing_current_user or skipped_candidates:
        print(
            "Conversation history best-effort summary: "
            f"missing_prev_user={missing_prev_user}, "
            f"missing_prev_choice={missing_prev_choice}, "
            f"missing_current_user={missing_current_user}, "
            f"skipped_candidate_turns={skipped_candidates}, "
            f"conversations_without_turns={skipped_turns}"
        )
    return pairs


def build_pairs(
    utterances: pd.DataFrame,
    system_prompt: Optional[str],
    dataset_format: str,
    delta: float = 0.0,
) -> List[dict]:
    """Construct (prompt, chosen, rejected) pairs per interaction."""
    pairs: List[dict] = []
    grouped = utterances.groupby("interaction_id")
    for interaction_id, group in tqdm(grouped, desc="Building pairs"):
        prompt = group["user_prompt"].iloc[0]
        user_id = group["user_id"].iloc[0]
        responses = group[["model_response", "score"]].dropna(subset=["model_response", "score"])
        if len(responses) < 2:
            continue
        resp_list = list(responses.itertuples(index=False, name=None))
        for i in range(len(resp_list)):
            for j in range(i + 1, len(resp_list)):
                resp_i, score_i = resp_list[i]
                resp_j, score_j = resp_list[j]
                try:
                    score_i = float(score_i)
                    score_j = float(score_j)
                except Exception:
                    continue
                diff = score_i - score_j
                if diff == 0:
                    continue
                if diff > 0 and diff >= delta:
                    chosen_resp = resp_i
                    rejected_resp = resp_j
                elif diff < 0 and (-diff) >= delta:
                    chosen_resp = resp_j
                    rejected_resp = resp_i
                else:
                    continue
                prompt_value, chosen_value, rejected_value = format_pair(
                    prompt=prompt,
                    chosen=chosen_resp,
                    rejected=rejected_resp,
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


def attach_weights(
    pairs: List[dict],
    weights: pd.Series | None,
    default_weight: float = 1.0,
) -> List[dict]:
    if weights is None:
        for p in pairs:
            p["weight"] = float(default_weight)
        return pairs
    weight_map = weights.to_dict()
    for p in pairs:
        p["weight"] = float(weight_map.get(p["user_id"], default_weight))
    return pairs


def normalize_per_rater(pairs: List[dict]) -> List[dict]:
    counts: dict = {}
    for p in pairs:
        user_id = p.get("user_id")
        counts[user_id] = counts.get(user_id, 0) + 1
    for p in pairs:
        user_id = p.get("user_id")
        denom = counts.get(user_id, 1)
        p["weight"] = float(p.get("weight", 1.0)) / float(denom)
    return pairs


def normalize_global_mean(pairs: List[dict]) -> List[dict]:
    if not pairs:
        return pairs
    weights = [float(p.get("weight", 1.0)) for p in pairs]
    sum_weights = sum(weights)
    if sum_weights <= 0:
        return pairs
    nonzero = sum(1 for w in weights if w > 0)
    target = nonzero if nonzero > 0 else len(weights)
    scale = float(target) / float(sum_weights)
    for p in pairs:
        p["weight"] = float(p.get("weight", 1.0)) * scale
    return pairs


def drop_zero_weight(pairs: List[dict]) -> List[dict]:
    if not pairs:
        return pairs
    kept = [p for p in pairs if float(p.get("weight", 0.0)) > 0.0]
    dropped = len(pairs) - len(kept)
    if dropped:
        print(f"Warning: dropped {dropped} zero-weight pairs before normalization")
    return kept


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
    rater_normalization: bool,
    conversations_df: Optional[pd.DataFrame] = None,
    use_conversations: bool = False,
    delta: float = 0.0,
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
    if use_conversations and conversations_df is not None:
        conv_filtered = conversations_df[conversations_df["user_id"].isin(panel_ids)]
        pairs = build_pairs_from_conversations(conv_filtered, filtered, delta, system_prompt, dataset_format)
    else:
        pairs = build_pairs(filtered, system_prompt, dataset_format, delta)
    pairs = attach_weights(pairs, None)
    if rater_normalization:
        pairs = normalize_per_rater(pairs)
    return pairs


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
    rater_normalization: bool,
    conversations_df: Optional[pd.DataFrame] = None,
    use_conversations: bool = False,
    delta: float = 0.0,
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
    filtered = utterances_df
    if use_conversations and conversations_df is not None:
        pairs = build_pairs_from_conversations(conversations_df, filtered, delta, system_prompt, dataset_format)
    else:
        pairs = build_pairs(filtered, system_prompt, dataset_format, delta)
    pairs = attach_weights(pairs, weights=weights, default_weight=0.0)
    pairs = drop_zero_weight(pairs)
    if rater_normalization:
        pairs = normalize_per_rater(pairs)
    return pairs


def prepare_us_rep(
    survey_df: pd.DataFrame,
    utterances_df: pd.DataFrame,
    system_prompt: Optional[str],
    dataset_format: str,
    conversations_df: Optional[pd.DataFrame] = None,
    use_conversations: bool = False,
    delta: float = 0.0,
) -> List[dict]:
    ids = survey_df.loc[survey_df["included_in_US_REP"] == True, "user_id"]
    filtered = utterances_df[utterances_df["user_id"].isin(ids)]
    if use_conversations and conversations_df is not None:
        conv_filtered = conversations_df[conversations_df["user_id"].isin(ids)]
        pairs = build_pairs_from_conversations(conv_filtered, filtered, delta, system_prompt, dataset_format)
    else:
        pairs = build_pairs(filtered, system_prompt, dataset_format, delta)
    return attach_weights(pairs, None)


def prepare_full(
    survey_df: pd.DataFrame,
    utterances_df: pd.DataFrame,
    system_prompt: Optional[str],
    dataset_format: str,
    conversations_df: Optional[pd.DataFrame] = None,
    use_conversations: bool = False,
    delta: float = 0.0,
) -> List[dict]:
    """Use all raters / utterances without filtering or weighting."""
    if use_conversations and conversations_df is not None:
        pairs = build_pairs_from_conversations(conversations_df, utterances_df, delta, system_prompt, dataset_format)
    else:
        pairs = build_pairs(utterances_df, system_prompt, dataset_format, delta)
    return attach_weights(pairs, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DPO datasets from PRISM.")
    parser.add_argument("--survey", type=Path, default=Path("prism-alignment/survey.jsonl"))
    parser.add_argument("--utterances", type=Path, default=Path("prism-alignment/utterances.jsonl"))
    parser.add_argument(
        "--conversations",
        type=Path,
        default=Path("prism-alignment/conversations.jsonl"),
        help="Conversation-level JSONL used to build multi-turn history.",
    )
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
        default="leximin",
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
    parser.add_argument(
        "--use-conversations",
        action="store_true",
        default=True,
        help="Use conversations.jsonl to build full multi-turn history per turn.",
    )
    parser.add_argument(
        "--no-use-conversations",
        action="store_false",
        dest="use_conversations",
        help="Disable multi-turn conversation history; use utterances only.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.0,
        help="Score margin threshold for pairwise preferences at each turn.",
    )
    parser.add_argument(
        "--rater-normalization",
        choices=["none", "panel", "all"],
        default="panel",
        help="Normalize per-rater contributions by dividing each example weight by the number of "
        "pairs from that rater. 'panel' applies only to hard/soft modes; 'all' applies to all modes.",
    )
    parser.add_argument("--system-prompt", default=None, help="Optional system prompt for chat format.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    survey_df = normalize_survey(load_jsonl(args.survey))
    utterances_df = normalize_utterances(load_jsonl(args.utterances))
    conversations_df = None
    if args.use_conversations:
        conversations_df = normalize_conversations(load_jsonl(args.conversations))
    print(
        "Loaded inputs: "
        f"survey_rows={len(survey_df)}, "
        f"utterance_rows={len(utterances_df)}, "
        f"conversation_rows={len(conversations_df) if conversations_df is not None else 0}"
    )
    print(
        "Preparation config: "
        f"mode={args.mode}, "
        f"use_conversations={args.use_conversations}, "
        f"dataset_format={args.dataset_format}, "
        f"delta={args.delta}, "
        f"rater_normalization={args.rater_normalization}"
    )

    rater_ids = set(utterances_df["user_id"].dropna().astype(str).tolist())
    if rater_ids:
        before = len(survey_df)
        survey_df = survey_df[survey_df["user_id"].isin(rater_ids)].copy()
        dropped = before - len(survey_df)
        if dropped:
            print(
                f"Filtered {dropped} survey-only rows without preference data "
                f"(kept {len(survey_df)} raters with utterances)."
            )
        if conversations_df is not None and "user_id" in conversations_df.columns:
            conv_before = len(conversations_df)
            conversations_df = conversations_df[
                conversations_df["user_id"].astype(str).isin(rater_ids)
            ].copy()
            conv_dropped = conv_before - len(conversations_df)
            if conv_dropped:
                print(
                    f"Filtered {conv_dropped} conversations from survey-only raters "
                    f"(kept {len(conversations_df)})."
                )

    if args.mode in {"hard", "soft"}:
        panel_config = load_panel_config(args.panel_config)
        print(
            "Panel config: "
            f"path={args.panel_config}, "
            f"algorithm={args.panel_algorithm}, "
            f"num_workers={args.num_workers}, "
            f"num_panel_samples={args.num_panel_samples if args.mode == 'soft' else 'n/a'}"
        )
    else:
        panel_config = None  # type: ignore

    normalize_panel = args.rater_normalization in {"panel", "all"}
    normalize_all = args.rater_normalization == "all"

    if args.mode == "hard":
        print("Starting hard-panel preparation.")
        records = prepare_hard_panel(
            survey_df=survey_df,
            utterances_df=utterances_df,
            panel_config=panel_config,
            panel_seed=args.panel_seed,
            panel_algorithm=args.panel_algorithm,
            system_prompt=args.system_prompt,
            dataset_format=args.dataset_format,
            rater_normalization=normalize_panel,
            conversations_df=conversations_df,
            use_conversations=args.use_conversations,
            delta=args.delta,
        )
    elif args.mode == "soft":
        print("Starting soft-panel preparation.")
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
            rater_normalization=normalize_panel,
            conversations_df=conversations_df,
            use_conversations=args.use_conversations,
            delta=args.delta,
        )
    else:
        print("Starting US-representative/full preparation.")
        records = prepare_us_rep(
            survey_df=survey_df,
            utterances_df=utterances_df,
            system_prompt=args.system_prompt,
            dataset_format=args.dataset_format,
            conversations_df=conversations_df,
            use_conversations=args.use_conversations,
            delta=args.delta,
        )
    if args.mode == "full":
        print("Switching to full-dataset preparation.")
        records = prepare_full(
            survey_df=survey_df,
            utterances_df=utterances_df,
            system_prompt=args.system_prompt,
            dataset_format=args.dataset_format,
            conversations_df=conversations_df,
            use_conversations=args.use_conversations,
            delta=args.delta,
        )
    if normalize_all and args.mode in {"full", "us_rep"}:
        records = normalize_per_rater(records)

    # Global normalization to keep average weight ~1 across contributing examples
    if records and "weight" in records[0]:
        records = normalize_global_mean(records)

    save_jsonl(records, args.output)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Convert the Community Alignment CSV into PRISM-like survey/utterances/conversations JSONL files.

The converted files are intentionally shaped to plug into the existing prepare_data.py pipeline:
- survey.jsonl: one row per annotator
- utterances.jsonl: one row per candidate response
- conversations.jsonl: one row per conversation with chosen-path history

Later turns are only retained along the longest contiguous valid prefix of the conversation. This
avoids constructing histories that skip over missing turns or missing candidate sets.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


TURN_PREFIXES = ["first", "second", "third", "fourth"]
RESPONSE_LABELS = ["response_a", "response_b", "response_c", "response_d"]
INVALID_REASONS = [
    "missing_prompt",
    "missing_preferred_response",
    "missing_candidate_response",
    "all_candidates_identical_after_dedup",
]


def clean_text(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.upper() == "EMPTY STRING":
        return None
    return text


def to_bool(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n", ""}:
        return False
    return bool(value)


def canonical_mode(values: Iterable[Optional[str]]) -> Optional[str]:
    counts: dict[str, int] = {}
    order: list[str] = []
    for value in values:
        if value is None:
            continue
        if value not in counts:
            counts[value] = 0
            order.append(value)
        counts[value] += 1
    if not counts:
        return None
    best = max(counts.values())
    for value in order:
        if counts[value] == best:
            return value
    return None


def normalize_country(value: object) -> Optional[str]:
    text = clean_text(value)
    if text is None:
        return None
    return re.sub(r"\s+", " ", text).strip().lower()


def slugify_locale(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or None


def normalize_age(value: object) -> Optional[str]:
    text = clean_text(value)
    if text is None:
        return None
    mapping = {
        "18-34": "18-34",
        "35-45": "35-45",
        "46-54": "46-54",
        "55+": "55+",
    }
    return mapping.get(text)


def normalize_gender(value: object) -> str:
    text = clean_text(value)
    if text is None:
        return "Prefer not to say"
    lowered = text.lower()
    if lowered == "male":
        return "Male"
    if lowered == "female":
        return "Female"
    if lowered == "other":
        return "Other"
    return "Prefer not to say"


def normalize_education(value: object) -> str:
    text = clean_text(value)
    if text is None:
        return "Prefer not to say"
    mapping = {
        "(At most) Complete Secondary": "(At most) Complete Secondary",
        "Some post-secondary": "Some post-secondary",
        "Post-secondary graduate": "Post-secondary graduate",
        "Some or complete graduate degree": "Some or complete graduate degree",
        "Other": "Other",
    }
    return mapping.get(text, "Prefer not to say")


def normalize_ethnicity(value: object) -> str:
    text = clean_text(value)
    if text is None:
        return "Prefer not to say"
    mapping = {
        "White": "White",
        "Black or African American": "Black or African American",
        "Hispanic or Latino": "Hispanic or Latino",
        "Asian": "Asian",
        "Other": "Other",
        "Prefer not to say": "Prefer not to say",
    }
    return mapping.get(text, "Prefer not to say")


def normalize_political(value: object) -> str:
    text = clean_text(value)
    if text is None:
        return "Prefer not to say"
    if text in {"Somewhat left-leaning", "Very left-leaning"}:
        return "Liberal"
    if text == "Middle-of-the-road, centrist":
        return "Moderate"
    if text in {"Somewhat right-leaning", "Very right-leaning"}:
        return "Conservative"
    if text == "I don't think of myself in this way":
        return "No identification"
    if text == "Prefer not to say":
        return "Prefer not to say"
    return "Prefer not to say"


def save_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def init_reason_counts() -> dict[str, int]:
    return {reason: 0 for reason in INVALID_REASONS}


def init_prefix_counts() -> dict[str, int]:
    return {str(i): 0 for i in range(len(TURN_PREFIXES) + 1)}


def build_survey(df: pd.DataFrame, us_country_name: str) -> list[dict]:
    survey_rows: list[dict] = []
    us_country = us_country_name.strip().lower()
    for annotator_id, group in df.groupby("annotator_id", sort=False):
        countries = [normalize_country(v) for v in group["annotator_country"].tolist()]
        country = canonical_mode(countries)
        study_locale = "us" if country == us_country else slugify_locale(country)
        survey_rows.append(
            {
                "user_id": str(annotator_id),
                "annotator_id": str(annotator_id),
                "source_dataset": "community_alignment",
                "annotator_country": country,
                "study_locale": study_locale,
                "included_in_US_REP": country == us_country,
                "survey_only": False,
                "age": canonical_mode(normalize_age(v) for v in group["annotator_age"].tolist()),
                "gender": canonical_mode(normalize_gender(v) for v in group["annotator_gender"].tolist()),
                "education": canonical_mode(
                    normalize_education(v) for v in group["annotator_education_level"].tolist()
                ),
                "ethnicity": canonical_mode(
                    normalize_ethnicity(v) for v in group["annotator_ethnicity"].tolist()
                ),
                "political": canonical_mode(
                    normalize_political(v) for v in group["annotator_political"].tolist()
                ),
            }
        )
    return survey_rows


def extract_turn(row: pd.Series, prefix: str) -> tuple[Optional[dict], Optional[str], dict[str, int]]:
    prompt = clean_text(row.get(f"{prefix}_turn_prompt"))
    if prompt is None:
        return None, "missing_prompt", {"duplicate_candidates_removed": 0}
    preferred_response = clean_text(row.get(f"{prefix}_turn_preferred_response"))
    if preferred_response not in RESPONSE_LABELS:
        return None, "missing_preferred_response", {"duplicate_candidates_removed": 0}
    responses = {
        label: clean_text(row.get(f"{prefix}_turn_{label}"))
        for label in RESPONSE_LABELS
    }
    if any(text is None for text in responses.values()):
        return None, "missing_candidate_response", {"duplicate_candidates_removed": 0}

    # Preserve the chosen response when duplicates occur, and collapse duplicate texts
    # among the remaining candidates instead of dropping the whole turn.
    ordered_labels = [preferred_response] + [label for label in RESPONSE_LABELS if label != preferred_response]
    seen_texts: set[str] = set()
    candidates: list[dict[str, object]] = []
    duplicates_removed = 0
    for label in ordered_labels:
        text = responses[label]
        assert text is not None
        if text in seen_texts:
            duplicates_removed += 1
            continue
        seen_texts.add(text)
        candidates.append(
            {
                "label": label,
                "text": text,
                "if_chosen": label == preferred_response,
            }
        )

    if len(candidates) < 2:
        return None, "all_candidates_identical_after_dedup", {"duplicate_candidates_removed": duplicates_removed}

    return (
        {
            "prompt": prompt,
            "preferred_response": preferred_response,
            "candidates": candidates,
            "feedback": clean_text(row.get(f"{prefix}_turn_feedback")),
        },
        None,
        {"duplicate_candidates_removed": duplicates_removed},
    )


def build_conversations_and_utterances(
    df: pd.DataFrame, us_country_name: str
) -> tuple[list[dict], list[dict], dict[str, object]]:
    conversations: list[dict] = []
    utterances: list[dict] = []
    us_country = us_country_name.strip().lower()
    stats: dict[str, object] = {
        "rows_total": int(len(df)),
        "rows_total_us": 0,
        "rows_with_kept_turns": 0,
        "rows_with_kept_turns_us": 0,
        "rows_dropped_entirely": 0,
        "rows_dropped_entirely_us": 0,
        "rows_truncated_after_gap": 0,
        "rows_truncated_after_gap_us": 0,
        "turns_kept": 0,
        "turns_kept_us": 0,
        "turns_dropped_after_gap": 0,
        "turns_dropped_after_gap_us": 0,
        "turns_salvaged_by_dedup": 0,
        "turns_salvaged_by_dedup_us": 0,
        "duplicate_candidates_removed": 0,
        "duplicate_candidates_removed_us": 0,
        "invalid_reasons_overall": init_reason_counts(),
        "invalid_reasons_us": init_reason_counts(),
        "prefix_lengths_overall": init_prefix_counts(),
        "prefix_lengths_us": init_prefix_counts(),
    }

    for row in df.to_dict(orient="records"):
        country = normalize_country(row.get("annotator_country"))
        is_us = country == us_country
        if is_us:
            stats["rows_total_us"] = int(stats["rows_total_us"]) + 1

        extracted_turns: list[Optional[dict]] = []
        invalid_reasons: list[Optional[str]] = []
        turn_metas: list[dict[str, int]] = []
        for prefix in TURN_PREFIXES:
            turn, reason, meta = extract_turn(pd.Series(row), prefix)
            extracted_turns.append(turn)
            invalid_reasons.append(reason)
            turn_metas.append(meta)

        valid_turns: list[dict] = []
        for turn in extracted_turns:
            if turn is None:
                break
            valid_turns.append(turn)

        prefix_len = len(valid_turns)
        prefix_lengths_overall = stats["prefix_lengths_overall"]
        assert isinstance(prefix_lengths_overall, dict)
        prefix_lengths_overall[str(prefix_len)] += 1
        if is_us:
            prefix_lengths_us = stats["prefix_lengths_us"]
            assert isinstance(prefix_lengths_us, dict)
            prefix_lengths_us[str(prefix_len)] += 1

        invalid_reasons_overall = stats["invalid_reasons_overall"]
        invalid_reasons_us = stats["invalid_reasons_us"]
        assert isinstance(invalid_reasons_overall, dict)
        assert isinstance(invalid_reasons_us, dict)
        for reason in invalid_reasons:
            if reason is None:
                continue
            invalid_reasons_overall[reason] += 1
            if is_us:
                invalid_reasons_us[reason] += 1

        all_valid_flags = [turn is not None for turn in extracted_turns]
        remaining_flags = all_valid_flags[prefix_len:]
        dropped_after_gap = sum(1 for flag in remaining_flags if flag)
        stats["turns_dropped_after_gap"] = int(stats["turns_dropped_after_gap"]) + dropped_after_gap
        if is_us:
            stats["turns_dropped_after_gap_us"] = int(stats["turns_dropped_after_gap_us"]) + dropped_after_gap
        if dropped_after_gap:
            stats["rows_truncated_after_gap"] = int(stats["rows_truncated_after_gap"]) + 1
            if is_us:
                stats["rows_truncated_after_gap_us"] = int(stats["rows_truncated_after_gap_us"]) + 1

        for turn, meta in zip(valid_turns, turn_metas[:prefix_len]):
            duplicates_removed = meta.get("duplicate_candidates_removed", 0)
            stats["duplicate_candidates_removed"] = int(stats["duplicate_candidates_removed"]) + duplicates_removed
            if is_us:
                stats["duplicate_candidates_removed_us"] = int(stats["duplicate_candidates_removed_us"]) + duplicates_removed
            if duplicates_removed > 0:
                stats["turns_salvaged_by_dedup"] = int(stats["turns_salvaged_by_dedup"]) + 1
                if is_us:
                    stats["turns_salvaged_by_dedup_us"] = int(stats["turns_salvaged_by_dedup_us"]) + 1

        if not valid_turns:
            stats["rows_dropped_entirely"] = int(stats["rows_dropped_entirely"]) + 1
            if is_us:
                stats["rows_dropped_entirely_us"] = int(stats["rows_dropped_entirely_us"]) + 1
            continue

        stats["rows_with_kept_turns"] = int(stats["rows_with_kept_turns"]) + 1
        stats["turns_kept"] = int(stats["turns_kept"]) + len(valid_turns)
        if is_us:
            stats["rows_with_kept_turns_us"] = int(stats["rows_with_kept_turns_us"]) + 1
            stats["turns_kept_us"] = int(stats["turns_kept_us"]) + len(valid_turns)

        conversation_id = str(row["conversation_id"])
        user_id = str(row["annotator_id"])
        chosen_history: list[dict] = []

        for turn_idx, turn in enumerate(valid_turns):
            chosen_history.append({"turn": turn_idx, "role": "user", "content": turn["prompt"]})
            chosen_label = turn["preferred_response"]
            chosen_idx = 0
            for within_turn_id, candidate in enumerate(turn["candidates"]):
                label = str(candidate["label"])
                text = str(candidate["text"])
                is_chosen = bool(candidate["if_chosen"])
                if is_chosen:
                    chosen_idx = within_turn_id
                utterances.append(
                    {
                        "utterance_id": f"{conversation_id}:{turn_idx}:{within_turn_id}",
                        "interaction_id": f"{conversation_id}:{turn_idx}",
                        "conversation_id": conversation_id,
                        "user_id": user_id,
                        "turn": turn_idx,
                        "within_turn_id": within_turn_id,
                        "conversation_type": "community_alignment",
                        "user_prompt": turn["prompt"],
                        "model_response": text,
                        "model_name": label,
                        "model_provider": "community_alignment",
                        "score": 1.0 if is_chosen else 0.0,
                        "if_chosen": is_chosen,
                        "included_in_balanced_subset": to_bool(row.get("in_balanced_subset", False)),
                    }
                )

            chosen_candidate = next(candidate for candidate in turn["candidates"] if candidate["if_chosen"])
            chosen_history.append(
                {
                    "turn": turn_idx,
                    "role": "assistant",
                    "content": chosen_candidate["text"],
                    "model_provider": "community_alignment",
                    "model_name": chosen_label,
                    "score": 1.0,
                    "if_chosen": True,
                    "within_turn_id": chosen_idx,
                }
            )

        conversations.append(
            {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "conversation_type": "community_alignment",
                "opening_prompt": valid_turns[0]["prompt"],
                "conversation_turns": len(valid_turns),
                "conversation_history": chosen_history,
                "assigned_lang": clean_text(row.get("assigned_lang")),
                "wave": row.get("wave"),
                "annotator_country": normalize_country(row.get("annotator_country")),
                "is_pregenerated_first_prompt": to_bool(row.get("is_pregenerated_first_prompt", False)),
                "included_in_balanced_subset": to_bool(row.get("in_balanced_subset", False)),
                "included_in_balanced_subset_10": to_bool(row.get("in_balanced_subset_10", False)),
            }
        )

    return conversations, utterances, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Community Alignment CSV into PRISM-like JSONL files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("community-alignment-dataset/community_alignment.csv"),
        help="Path to community_alignment.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/community_alignment/normalized"),
        help="Directory for survey/utterances/conversations JSONL outputs.",
    )
    parser.add_argument(
        "--us-country-name",
        default="united states",
        help="Country value to treat as the U.S. subset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    survey = build_survey(df, us_country_name=args.us_country_name)
    conversations, utterances, stats = build_conversations_and_utterances(df, us_country_name=args.us_country_name)

    output_dir = args.output_dir
    save_jsonl(survey, output_dir / "survey.jsonl")
    save_jsonl(conversations, output_dir / "conversations.jsonl")
    save_jsonl(utterances, output_dir / "utterances.jsonl")
    stats_path = output_dir / "conversion_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(survey)} survey rows to {output_dir / 'survey.jsonl'}")
    print(f"Wrote {len(conversations)} conversations to {output_dir / 'conversations.jsonl'}")
    print(f"Wrote {len(utterances)} utterances to {output_dir / 'utterances.jsonl'}")
    print(f"Wrote conversion stats to {stats_path}")
    print(
        "Overall conversations: "
        f"total={stats['rows_total']}, "
        f"kept={stats['rows_with_kept_turns']}, "
        f"dropped_entirely={stats['rows_dropped_entirely']}, "
        f"truncated_after_gap={stats['rows_truncated_after_gap']}"
    )
    print(
        "US conversations: "
        f"total={stats['rows_total_us']}, "
        f"kept={stats['rows_with_kept_turns_us']}, "
        f"dropped_entirely={stats['rows_dropped_entirely_us']}, "
        f"truncated_after_gap={stats['rows_truncated_after_gap_us']}"
    )
    print(
        "Turn retention: "
        f"turns_kept={stats['turns_kept']}, "
        f"turns_kept_us={stats['turns_kept_us']}, "
        f"turns_dropped_after_gap={stats['turns_dropped_after_gap']}, "
        f"turns_dropped_after_gap_us={stats['turns_dropped_after_gap_us']}"
    )
    print(
        "Dedup salvage: "
        f"turns_salvaged_by_dedup={stats['turns_salvaged_by_dedup']}, "
        f"turns_salvaged_by_dedup_us={stats['turns_salvaged_by_dedup_us']}, "
        f"duplicate_candidates_removed={stats['duplicate_candidates_removed']}, "
        f"duplicate_candidates_removed_us={stats['duplicate_candidates_removed_us']}"
    )
    print(f"Contiguous prefix lengths (overall): {json.dumps(stats['prefix_lengths_overall'], sort_keys=True)}")
    print(f"Contiguous prefix lengths (US): {json.dumps(stats['prefix_lengths_us'], sort_keys=True)}")
    print(f"Invalid turn reasons (overall): {json.dumps(stats['invalid_reasons_overall'], sort_keys=True)}")
    print(f"Invalid turn reasons (US): {json.dumps(stats['invalid_reasons_us'], sort_keys=True)}")


if __name__ == "__main__":
    main()

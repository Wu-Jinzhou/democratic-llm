#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def clean_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.upper() == "EMPTY STRING":
        return None
    return text


def save_jsonl(records: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter Community Alignment normalized JSONL files by assigned conversation language."
    )
    parser.add_argument("--survey", type=Path, required=True)
    parser.add_argument("--utterances", type=Path, required=True)
    parser.add_argument("--conversations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--assigned-lang",
        default="en",
        help="Keep only conversations whose assigned_lang matches this value (case-insensitive).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_lang = args.assigned_lang.strip().lower()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    kept_conversation_ids: set[str] = set()
    kept_user_ids: set[str] = set()
    user_langs: dict[str, set[str]] = defaultdict(set)
    lang_counts: Counter[str] = Counter()
    total_conversations = 0

    kept_conversations_path = output_dir / "conversations.jsonl"
    with kept_conversations_path.open("w", encoding="utf-8") as out_f:
        for record in load_jsonl(args.conversations):
            total_conversations += 1
            conversation_id = str(record["conversation_id"])
            user_id = str(record["user_id"])
            assigned_lang = clean_text(record.get("assigned_lang"))
            normalized_lang = assigned_lang.lower() if assigned_lang is not None else ""
            user_langs[user_id].add(normalized_lang or "<missing>")
            lang_counts[normalized_lang or "<missing>"] += 1
            if normalized_lang != target_lang:
                continue
            kept_conversation_ids.add(conversation_id)
            kept_user_ids.add(user_id)
            out_f.write(json.dumps(record, ensure_ascii=False))
            out_f.write("\n")

    multilingual_kept_users = sum(
        1 for user_id in kept_user_ids if len(user_langs[user_id] - {"<missing>"}) > 1
    )

    kept_utterances = save_jsonl(
        (
            record
            for record in load_jsonl(args.utterances)
            if str(record.get("conversation_id")) in kept_conversation_ids
        ),
        output_dir / "utterances.jsonl",
    )
    kept_survey = save_jsonl(
        (
            record
            for record in load_jsonl(args.survey)
            if str(record.get("user_id")) in kept_user_ids
        ),
        output_dir / "survey.jsonl",
    )

    stats = {
        "assigned_lang": target_lang,
        "total_conversations": total_conversations,
        "kept_conversations": len(kept_conversation_ids),
        "kept_utterances": kept_utterances,
        "kept_survey_rows": kept_survey,
        "kept_users": len(kept_user_ids),
        "multilingual_kept_users": multilingual_kept_users,
        "language_counts": dict(sorted(lang_counts.items())),
    }
    stats_path = output_dir / "filter_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Filtered conversations to assigned_lang={target_lang}")
    print(f"Wrote {len(kept_conversation_ids)} conversations to {kept_conversations_path}")
    print(f"Wrote {kept_utterances} utterances to {output_dir / 'utterances.jsonl'}")
    print(f"Wrote {kept_survey} survey rows to {output_dir / 'survey.jsonl'}")
    print(f"Kept users={len(kept_user_ids)}, multilingual_kept_users={multilingual_kept_users}")
    print(f"Wrote filter stats to {stats_path}")


if __name__ == "__main__":
    main()

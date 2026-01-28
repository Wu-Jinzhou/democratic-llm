#!/usr/bin/env python3
"""
Panel feasibility + representativeness report.

This script:
1) Loads PRISM `survey.jsonl`
2) Applies `configs/*_config.yaml` to define demographic quotas + tolerance
3) Checks necessary feasibility conditions (enough people in each category to meet mins)
4) Samples one panel using LEGACY/LEXIMIN/RANDOM
5) Reports realized panel proportions vs targets and quota bounds

Outputs a single JSON file suitable for paper appendix tables.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Ensure local imports work even when executed as a script
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from sortition import (  # type: ignore
    PanelConfig,
    check_panel_feasibility,
    load_panel_config,
    prepare_panel_data,
    sample_panel,
)

# We intentionally use these internal helpers to ensure normalization matches the
# sampling implementation.
from sortition import _bounds_for_attribute, _build_stratification_inputs  # type: ignore


def _looks_like_git_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("rb") as f:
        header = f.read(200)
    return b"git-lfs" in header and b"version https://git-lfs.github.com/spec/v1" in header


def load_jsonl(path: Path) -> pd.DataFrame:
    if _looks_like_git_lfs_pointer(path):
        raise RuntimeError(
            f"{path} looks like a Git LFS pointer file. Run `git lfs pull` in the dataset repo."
        )
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        records: List[dict] = []
        bad = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    bad += 1
        if bad:
            print(f"Warning: skipped {bad} malformed lines in {path}")
        return pd.DataFrame.from_records(records)


def _counts_from_people(
    people: Dict[str, Dict[str, str]], attrs: List[str], ids: Optional[set[str]] = None
) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {a: {} for a in attrs}
    if ids is None:
        selected_people = people.values()
    else:
        selected_people = (people[i] for i in ids if i in people)
    for person in selected_people:
        for a in attrs:
            v = person.get(a)
            if v is None:
                continue
            out[a][v] = out[a].get(v, 0) + 1
    return out


def _panel_attr_rows(
    config: PanelConfig,
    pool_counts: Dict[str, Dict[str, int]],
    panel_counts: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    attrs_report: Dict[str, Any] = {}
    k = config.panel_size
    for attr in config.attributes:
        bounds = _bounds_for_attribute(attr, k)
        rows = []
        for category, target_p in attr.population_proportions.items():
            pool_n = int(pool_counts.get(attr.name, {}).get(category, 0))
            panel_n = int(panel_counts.get(attr.name, {}).get(category, 0))
            panel_p = (panel_n / k) if k > 0 else 0.0
            lower, upper = bounds.get(category, (0, k))
            rows.append(
                {
                    "category": category,
                    "target_proportion": float(target_p),
                    "tolerance": float(attr.tolerance),
                    "slack": bool(category in attr.slack_categories),
                    "bounds_count": {"min": int(lower), "max": int(upper)},
                    "pool_count": pool_n,
                    "panel_count": panel_n,
                    "panel_proportion": float(panel_p),
                    "delta_proportion": float(panel_p - target_p),
                    "within_bounds": bool(lower <= panel_n <= upper),
                }
            )
        attrs_report[attr.name] = {
            "column": attr.column,
            "nested_key": attr.nested_key,
            "slack_categories": list(attr.slack_categories),
            "rows": rows,
        }
    return attrs_report


def build_report(
    survey_path: Path,
    utterances_path: Optional[Path],
    panel_config_path: Path,
    algorithm: str,
    panel_seed: int,
    max_attempts: int,
    include_user_ids: bool,
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    config = load_panel_config(panel_config_path)
    survey_df = load_jsonl(survey_path)
    if "user_id" not in survey_df.columns:
        raise KeyError(
            f"Survey missing required column 'user_id'. Available columns: {list(survey_df.columns)}"
        )
    survey_df["user_id"] = survey_df["user_id"].astype(str)

    rater_filter_count = 0
    if utterances_path is not None:
        utterances_df = load_jsonl(utterances_path)
        if "user_id" in utterances_df.columns:
            utter_ids = set(utterances_df["user_id"].astype(str).dropna().tolist())
            before = len(survey_df)
            survey_df = survey_df[survey_df["user_id"].isin(utter_ids)].copy()
            rater_filter_count = before - len(survey_df)
            if rater_filter_count:
                print(
                    f"Filtered {rater_filter_count} survey-only rows without preferences "
                    f"(kept {len(survey_df)} raters with utterances)."
                )

    prepared = prepare_panel_data(survey_df, config)
    feasibility_issues = check_panel_feasibility(prepared, config.attributes, config.panel_size)
    categories, people = _build_stratification_inputs(prepared, config.attributes, config.panel_size)
    pool_counts = _counts_from_people(people, [a.name for a in config.attributes], ids=None)

    sampled_panel: Optional[pd.DataFrame] = None
    sample_error: Optional[str] = None
    try:
        sampled_panel = sample_panel(
            prepared,
            attrs=config.attributes,
            panel_size=config.panel_size,
            algorithm=algorithm,
            max_attempts=max_attempts,
            rng=random.Random(panel_seed),
        )
    except Exception as exc:
        sample_error = str(exc)

    panel_counts: Dict[str, Dict[str, int]] = {a.name: {} for a in config.attributes}
    panel_user_ids: List[str] = []
    if sampled_panel is not None:
        panel_user_ids = [str(x) for x in sampled_panel["user_id"].tolist()]
        panel_id_set = set(panel_user_ids)
        panel_counts = _counts_from_people(people, [a.name for a in config.attributes], ids=panel_id_set)

    report: Dict[str, Any] = {
        "survey_path": str(survey_path),
        "utterances_path": str(utterances_path) if utterances_path else None,
        "panel_config_path": str(panel_config_path),
        "algorithm": algorithm,
        "panel_seed": int(panel_seed),
        "panel_size": int(config.panel_size),
        "locale_filter": config.locale_filter,
        "survey_only_removed": int(rater_filter_count),
        "pool_rows_after_locale_filter": int(len(prepared)),
        "pool_valid_people": int(len(people)),
        "feasibility_issues": [
            {"attribute": a, "category": c, "needed_min": int(need), "available": int(avail)}
            for a, c, need, avail in feasibility_issues
        ],
        "sample_error": sample_error,
        "sampled_panel": {
            "rows": int(len(sampled_panel)) if sampled_panel is not None else 0,
            "unique_user_ids": int(len(set(panel_user_ids))),
            **({"user_ids": panel_user_ids} if include_user_ids else {}),
        },
        "attribute_reports": _panel_attr_rows(config, pool_counts=pool_counts, panel_counts=panel_counts),
        "categories_snapshot": categories,  # useful for debugging quotas in appendix
    }
    return report, sampled_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Panel feasibility + representativeness report.")
    parser.add_argument("--survey", type=Path, default=Path("prism-alignment/survey.jsonl"))
    parser.add_argument(
        "--utterances",
        type=Path,
        default=Path("prism-alignment/utterances.jsonl"),
        help="Optional utterances JSONL to filter survey-only raters.",
    )
    parser.add_argument("--panel-config", type=Path, default=Path("configs/panel_config.yaml"))
    parser.add_argument(
        "--panel-algorithm",
        choices=["leximin", "legacy", "random"],
        default="leximin",
        help="Panel algorithm used to sample a single panel for reporting.",
    )
    parser.add_argument("--panel-seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=5000)
    parser.add_argument(
        "--include-user-ids",
        action="store_true",
        help="Include sampled panel user_ids in the JSON (off by default to avoid leaking identifiers).",
    )
    parser.add_argument(
        "--panel-ids-output",
        type=Path,
        default=None,
        help="Optional path to write sampled panel user_ids (one per line).",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, panel_df = build_report(
        survey_path=args.survey,
        utterances_path=args.utterances if args.utterances else None,
        panel_config_path=args.panel_config,
        algorithm=args.panel_algorithm,
        panel_seed=args.panel_seed,
        max_attempts=args.max_attempts,
        include_user_ids=bool(args.include_user_ids),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote panel report to {args.output}")

    if args.panel_ids_output is not None:
        if panel_df is None or panel_df.empty:
            raise RuntimeError(
                "No panel was sampled; cannot write panel ids. Check --panel-algorithm and inputs."
            )
        ids = [str(x) for x in panel_df["user_id"].tolist()]
        args.panel_ids_output.parent.mkdir(parents=True, exist_ok=True)
        args.panel_ids_output.write_text("\n".join(ids) + "\n")
        print(f"Wrote sampled panel user_ids to {args.panel_ids_output}")


if __name__ == "__main__":
    main()

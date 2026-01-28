#!/usr/bin/env python3
"""
Selection-probability (soft panel) diagnostics.

For LEXIMIN:
- Computes exact per-rater inclusion probabilities π_i under the lottery.
- Summarizes distributional diagnostics (concentration, Gini, ESS).
- Reports expected demographic composition induced by π_i.

Optionally compares π_i mass for raters that actually appear in the DPO dataset
(since some raters may have 0 usable preference pairs).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure local imports work even when executed as a script
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from sortition import (  # type: ignore
    estimate_selection_probabilities,
    load_panel_config,
    prepare_panel_data,
)

from sortition import _build_stratification_inputs  # type: ignore


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


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


def top_mass_share(x: np.ndarray, frac: float) -> float:
    x = np.asarray(x, dtype=float)
    total = float(x.sum())
    if total <= 0:
        return 0.0
    n = x.size
    k = max(1, int(np.ceil(n * frac)))
    return float(np.sort(x)[::-1][:k].sum() / total)


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    s1 = float(w.sum())
    s2 = float(np.square(w).sum())
    if s2 <= 0:
        return 0.0
    return (s1 * s1) / s2


def load_dataset_user_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = obj.get("user_id")
            if uid is not None:
                ids.add(str(uid))
    return ids


def expected_demographics(
    people: Dict[str, Dict[str, str]],
    pi_by_id: Dict[str, float],
    attributes: List[str],
    k: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for attr in attributes:
        by_cat: Dict[str, float] = {}
        for pid, person in people.items():
            v = person.get(attr)
            if v is None:
                continue
            by_cat[v] = by_cat.get(v, 0.0) + float(pi_by_id.get(pid, 0.0))
        rows = []
        for cat, exp_count in sorted(by_cat.items(), key=lambda kv: (-kv[1], kv[0])):
            rows.append(
                {
                    "category": cat,
                    "expected_count": float(exp_count),
                    "expected_proportion": float(exp_count / k) if k > 0 else 0.0,
                }
            )
        out[attr] = {"rows": rows}
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Soft-panel selection probability diagnostics.")
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
        help="Algorithm used to compute π_i. For LEXIMIN this is exact; for LEGACY/random it is sampled.",
    )
    parser.add_argument("--num-panel-samples", type=int, default=2000)
    parser.add_argument("--panel-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/data/soft_panel.jsonl"),
        help="Optional dataset JSONL to measure how much π-mass is actually used.",
    )
    parser.add_argument("--per-rater-output", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_panel_config(args.panel_config)
    survey_df = load_jsonl(args.survey)
    if "user_id" not in survey_df.columns:
        raise KeyError(
            f"Survey missing required column 'user_id'. Available columns: {list(survey_df.columns)}"
        )
    survey_df["user_id"] = survey_df["user_id"].astype(str)

    rater_filter_count = 0
    if args.utterances and args.utterances.exists():
        utterances_df = load_jsonl(args.utterances)
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

    pi_series = estimate_selection_probabilities(
        prepared,
        attrs=config.attributes,
        panel_size=config.panel_size,
        num_samples=args.num_panel_samples,
        rng_seed=args.panel_seed,
        num_workers=args.num_workers,
        algorithm=args.panel_algorithm,
    )
    user_ids = prepared["user_id"].astype(str).tolist()
    pi_values = pi_series.astype(float).to_numpy()
    pi_by_id = {uid: float(pi) for uid, pi in zip(user_ids, pi_values)}

    categories, people = _build_stratification_inputs(prepared, config.attributes, config.panel_size)
    valid_ids = list(people.keys())
    pi_valid = np.array([pi_by_id.get(pid, 0.0) for pid in valid_ids], dtype=float)

    k = int(config.panel_size)
    total_pi = float(pi_valid.sum())
    dataset_ids = load_dataset_user_ids(args.dataset) if args.dataset else set()
    used_pi_mass = float(sum(pi_by_id.get(pid, 0.0) for pid in dataset_ids))

    report: Dict[str, Any] = {
        "method": "selection-probabilities",
        "survey_path": str(args.survey),
        "utterances_path": str(args.utterances) if args.utterances else None,
        "panel_config_path": str(args.panel_config),
        "algorithm": args.panel_algorithm,
        "params": {
            "panel_seed": int(args.panel_seed),
            "num_panel_samples": int(args.num_panel_samples),
            "num_workers": int(args.num_workers),
        },
        "panel_size": k,
        "survey_only_removed": int(rater_filter_count),
        "pool_valid_people": int(len(valid_ids)),
        "sum_pi": total_pi,
        "sum_pi_expected": float(k),
        "pi_stats": {
            "min": float(np.min(pi_valid)) if pi_valid.size else 0.0,
            "p05": float(np.quantile(pi_valid, 0.05)) if pi_valid.size else 0.0,
            "median": float(np.median(pi_valid)) if pi_valid.size else 0.0,
            "mean": float(np.mean(pi_valid)) if pi_valid.size else 0.0,
            "p95": float(np.quantile(pi_valid, 0.95)) if pi_valid.size else 0.0,
            "max": float(np.max(pi_valid)) if pi_valid.size else 0.0,
            "std": float(np.std(pi_valid)) if pi_valid.size else 0.0,
            "gini": float(gini(pi_valid)),
            "ess": float(effective_sample_size(pi_valid)),
            "top_1pct_mass_share": float(top_mass_share(pi_valid, 0.01)),
            "top_5pct_mass_share": float(top_mass_share(pi_valid, 0.05)),
            "top_10pct_mass_share": float(top_mass_share(pi_valid, 0.10)),
        },
        "used_mass": {
            "dataset_path": str(args.dataset) if args.dataset else None,
            "unique_raters_in_dataset": int(len(dataset_ids)),
            "sum_pi_over_dataset_raters": used_pi_mass,
            "mass_share": (used_pi_mass / total_pi) if total_pi > 0 else 0.0,
        },
        "expected_demographics": expected_demographics(
            people=people,
            pi_by_id=pi_by_id,
            attributes=[a.name for a in config.attributes],
            k=k,
        ),
        "categories_snapshot": categories,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote selection-probability diagnostics to {args.output}")

    if args.per_rater_output is not None:
        args.per_rater_output.parent.mkdir(parents=True, exist_ok=True)
        out_df = pd.DataFrame(
            {"user_id": valid_ids, "pi": [pi_by_id.get(pid, 0.0) for pid in valid_ids]}
        )
        out_df.to_csv(args.per_rater_output, index=False)
        print(f"Wrote per-rater π_i to {args.per_rater_output}")


if __name__ == "__main__":
    main()

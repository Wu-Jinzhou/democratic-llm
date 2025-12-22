#!/usr/bin/env python3
"""
Sortition helpers for building demographic panels from PRISM survey data.

Supports:
- Hard panel selection: sample one panel using LEGACY or LEXIMIN.
- Soft panel weighting: estimate selection probabilities (LEXIMIN exact, LEGACY via Monte Carlo).

Configuration is driven by a YAML file with entries for each demographic attribute:
  attributes:
    - name: ethnicity
      column: ethnicity
      nested_key: simplified
      population_proportions:
        White: 0.6
        "Black, African American, or Afro-Caribbean": 0.12
        Asian: 0.06
        Hispanic or Latino: 0.18
        Other: 0.04
      tolerance: 0.05   # +/- 5% around target proportion
  panel_size: 300
"""
from __future__ import annotations

import copy
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


@dataclass
class AttributeConfig:
    name: str
    column: str
    nested_key: Optional[str]
    population_proportions: Dict[str, float]
    tolerance: float = 0.05
    slack_categories: List[str] = field(default_factory=list)


@dataclass
class PanelConfig:
    attributes: List[AttributeConfig]
    panel_size: int
    locale_filter: Optional[str] = None


def load_panel_config(path: Path) -> PanelConfig:
    data = yaml.safe_load(path.read_text())
    attributes = [
        AttributeConfig(
            name=entry["name"],
            column=entry["column"],
            nested_key=entry.get("nested_key"),
            population_proportions=entry["population_proportions"],
            tolerance=float(entry.get("tolerance", data.get("tolerance", 0.05))),
            slack_categories=list(entry.get("slack_categories", [])),
        )
        for entry in data["attributes"]
    ]
    return PanelConfig(
        attributes=attributes,
        panel_size=int(data["panel_size"]),
        locale_filter=data.get("locale_filter"),
    )


def _extract_value(row: pd.Series, attr: AttributeConfig):
    val = row.get(attr.column)
    if isinstance(val, dict) and attr.nested_key:
        return val.get(attr.nested_key)
    return val


def add_attribute_columns(df: pd.DataFrame, attrs: Iterable[AttributeConfig]) -> pd.DataFrame:
    """Adds flattened attribute columns for sortition."""
    out = df.copy()
    for attr in attrs:
        col_name = f"{attr.name}_flattened"
        out[col_name] = out.apply(lambda r: _extract_value(r, attr), axis=1)
    return out


def _bounds_for_attribute(attr: AttributeConfig, panel_size: int):
    bounds = {}
    for category, proportion in attr.population_proportions.items():
        if category in attr.slack_categories:
            bounds[category] = (0, panel_size)
            continue
        target = proportion * panel_size
        lower = max(0, int(np.floor(target * (1 - attr.tolerance))))
        upper = min(panel_size, int(np.ceil(target * (1 + attr.tolerance))))
        bounds[category] = (lower, upper)
    return bounds


def _normalize_value(value: Optional[str], attr: AttributeConfig) -> Optional[str]:
    if value in attr.population_proportions:
        return value
    # Treat missing/unknown values as Other or Prefer not to say when available
    if value is None or (isinstance(value, float) and np.isnan(value)):
        if "Prefer not to say" in attr.population_proportions:
            return "Prefer not to say"
    if "Other" in attr.population_proportions:
        return "Other"
    if "Prefer not to say" in attr.population_proportions:
        return "Prefer not to say"
    return None


def _build_stratification_inputs(
    df: pd.DataFrame, attrs: List[AttributeConfig], panel_size: int
) -> Tuple[Dict[str, Dict[str, Dict[str, int]]], Dict[str, Dict[str, str]]]:
    """Build categories + people dicts expected by Sortition Foundation algorithms."""
    categories: Dict[str, Dict[str, Dict[str, int]]] = {}
    for attr in attrs:
        bounds = _bounds_for_attribute(attr, panel_size)
        categories[attr.name] = {}
        for category, (lower, upper) in bounds.items():
            categories[attr.name][category] = {
                "name": category,
                "min": int(lower),
                "max": int(upper),
                "selected": 0,
                "remaining": 0,
            }

    people: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        person_id = str(row.get("user_id"))
        person: Dict[str, str] = {}
        valid = True
        for attr in attrs:
            raw = _extract_value(row, attr)
            value = _normalize_value(raw, attr)
            if value is None or value not in categories[attr.name]:
                valid = False
                break
            person[attr.name] = value
        if not valid:
            continue
        people[person_id] = person

    for person in people.values():
        for attr_name, value in person.items():
            categories[attr_name][value]["remaining"] += 1

    return categories, people


def check_panel_feasibility(
    df: pd.DataFrame, attrs: List[AttributeConfig], panel_size: int
) -> List[Tuple[str, str, int, int]]:
    """
    Return a list of unmet requirements of the form (attribute, category, needed, available).
    If the list is empty, the panel is feasible in principle given the dataset.
    """
    issues: List[Tuple[str, str, int, int]] = []
    for attr in attrs:
        col = f"{attr.name}_flattened"
        counts = df[col].value_counts(dropna=False).to_dict()
        bounds = _bounds_for_attribute(attr, panel_size)
        for category, (lower, _upper) in bounds.items():
            available = counts.get(category, 0)
            if available < lower:
                issues.append((attr.name, category, lower, available))
    return issues


def _legacy_delete_all_in_cat(categories, people, cat, cat_value):
    people_to_delete = []
    for pkey, person in people.items():
        if person[cat] == cat_value:
            people_to_delete.append(pkey)
            for pcat, pval in person.items():
                cat_item = categories[pcat][pval]
                cat_item["remaining"] -= 1
                if cat_item["remaining"] == 0 and cat_item["selected"] < cat_item["min"]:
                    raise RuntimeError(
                        f"LEGACY fail: not enough remaining in {pcat}:{pval} after delete."
                    )
    for p in people_to_delete:
        del people[p]
    return len(people_to_delete), len(people)


def _legacy_really_delete_person(categories, people, pkey, selected):
    for pcat, pval in people[pkey].items():
        cat_item = categories[pcat][pval]
        if selected:
            cat_item["selected"] += 1
        cat_item["remaining"] -= 1
        if cat_item["remaining"] == 0 and cat_item["selected"] < cat_item["min"]:
            raise RuntimeError(f"LEGACY fail: no one left in {pcat}:{pval}.")
    del people[pkey]


def _legacy_delete_person(categories, people, pkey):
    person = people[pkey]
    _legacy_really_delete_person(categories, people, pkey, True)
    for pcat, pval in person.items():
        cat_item = categories[pcat][pval]
        if cat_item["selected"] == cat_item["max"]:
            _legacy_delete_all_in_cat(categories, people, pcat, pval)


def _legacy_find_max_ratio_cat(categories, rng: random.Random):
    ratio = -100.0
    key_max = ""
    index_max_name = ""
    random_person_num = -1
    for cat_key, cats in categories.items():
        for cat, cat_item in cats.items():
            if cat_item["selected"] < cat_item["min"] and cat_item["remaining"] < (
                cat_item["min"] - cat_item["selected"]
            ):
                raise RuntimeError(
                    f"LEGACY fail: not enough remaining in {cat_key}:{cat}."
                )
            if cat_item["remaining"] != 0 and cat_item["max"] != 0:
                item_ratio = (cat_item["min"] - cat_item["selected"]) / float(
                    cat_item["remaining"]
                )
                if item_ratio > 1:
                    raise RuntimeError("LEGACY fail: ratio > 1.")
                if item_ratio > ratio:
                    ratio = item_ratio
                    key_max = cat_key
                    index_max_name = cat
                    random_person_num = rng.randint(1, cat_item["remaining"])
    return {
        "ratio_cat": key_max,
        "ratio_cat_val": index_max_name,
        "ratio_random": random_person_num,
    }


def _legacy_find_random_sample(
    categories: Dict[str, Dict[str, Dict[str, int]]],
    people: Dict[str, Dict[str, str]],
    number_people_wanted: int,
    rng: random.Random,
) -> Dict[str, Dict[str, str]]:
    people_selected: Dict[str, Dict[str, str]] = {}
    for count in range(number_people_wanted):
        ratio = _legacy_find_max_ratio_cat(categories, rng)
        for pkey, pvalue in list(people.items()):
            if pvalue[ratio["ratio_cat"]] == ratio["ratio_cat_val"]:
                ratio["ratio_random"] -= 1
                if ratio["ratio_random"] == 0:
                    people_selected.update({pkey: pvalue})
                    _legacy_delete_person(categories, people, pkey)
                    break
        if count < (number_people_wanted - 1) and len(people) == 0:
            raise RuntimeError("LEGACY fail: ran out of people.")
    return people_selected


def _load_stratification_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_dir = repo_root / "third_party"
    if not module_dir.exists():
        raise ImportError("third_party module not found in repo.")
    if str(module_dir) not in sys.path:
        sys.path.append(str(module_dir))
    import stratification  # type: ignore

    return stratification


def _leximin_distribution(
    categories: Dict[str, Dict[str, Dict[str, int]]],
    people: Dict[str, Dict[str, str]],
    panel_size: int,
) -> Tuple[List[frozenset], List[float]]:
    strat = _load_stratification_module()
    # LEXIMIN uses gurobi + mip; raise a clear error if missing
    try:
        committees, probabilities, _ = strat.find_distribution_leximin(
            categories=categories,
            people=people,
            columns_data={},
            number_people_wanted=panel_size,
            check_same_address=False,
            check_same_address_columns=[],
        )
    except Exception as exc:
        raise RuntimeError(
            "LEXIMIN requires gurobipy + python-mip; ensure they are installed and licensed."
        ) from exc
    return committees, probabilities


def _is_feasible(sampled: pd.DataFrame, attrs: List[AttributeConfig], panel_size: int) -> bool:
    for attr in attrs:
        col = f"{attr.name}_flattened"
        counts = sampled[col].value_counts(dropna=False).to_dict()
        bounds = _bounds_for_attribute(attr, panel_size)
        for category, (lower, upper) in bounds.items():
            count = counts.get(category, 0)
            if count < lower or count > upper:
                return False
    return True


def _sample_panel_rejection(
    df: pd.DataFrame,
    attrs: List[AttributeConfig],
    panel_size: int,
    max_attempts: int = 5000,
    rng: Optional[random.Random] = None,
) -> pd.DataFrame:
    """Randomly sample a panel that satisfies attribute bounds."""
    rng = rng or random.Random()
    df = df.reset_index(drop=True)
    indices = list(df.index)
    issues = check_panel_feasibility(df, attrs, panel_size)
    if issues:
        details = "; ".join(
            f"{attr}:{cat} needs {needed}, available {avail}"
            for attr, cat, needed, avail in issues
        )
        raise RuntimeError(f"Panel infeasible given current data: {details}")
    for _ in range(max_attempts):
        chosen_idx = rng.sample(indices, k=panel_size)
        sampled = df.loc[chosen_idx]
        if _is_feasible(sampled, attrs, panel_size):
            return sampled
    # If we reach here, sampling failed despite theoretical feasibility
    details = []
    for attr in attrs:
        col = f"{attr.name}_flattened"
        counts = df[col].value_counts(dropna=False).to_dict()
        bounds = _bounds_for_attribute(attr, panel_size)
        details.append(
            f"{attr.name}: bounds {bounds}, available {counts}"
        )
    raise RuntimeError(
        "Failed to sample a feasible panel after "
        f"{max_attempts} attempts; consider relaxing tolerances, lowering panel_size, "
        "or increasing max_attempts. Constraint snapshot: " + " | ".join(details)
    )


def sample_panel(
    df: pd.DataFrame,
    attrs: List[AttributeConfig],
    panel_size: int,
    algorithm: str = "legacy",
    max_attempts: int = 5000,
    rng: Optional[random.Random] = None,
) -> pd.DataFrame:
    """Sample a panel using LEGACY or LEXIMIN; fall back to rejection if requested."""
    rng = rng or random.Random()
    if algorithm == "legacy":
        categories, people = _build_stratification_inputs(df, attrs, panel_size)
        # Use a fresh copy since the legacy algorithm mutates the inputs
        selected = _legacy_find_random_sample(
            copy.deepcopy(categories),
            copy.deepcopy(people),
            panel_size,
            rng,
        )
        selected_ids = set(selected.keys())
        return df.loc[df["user_id"].astype(str).isin(selected_ids)]
    if algorithm == "leximin":
        categories, people = _build_stratification_inputs(df, attrs, panel_size)
        committees, probabilities = _leximin_distribution(
            copy.deepcopy(categories), copy.deepcopy(people), panel_size
        )
        if not committees:
            raise RuntimeError("LEXIMIN returned no feasible committees.")
        chosen = rng.choices(committees, weights=probabilities, k=1)[0]
        return df.loc[df["user_id"].astype(str).isin(chosen)]
    if algorithm == "random":
        return _sample_panel_rejection(df, attrs, panel_size, max_attempts=max_attempts, rng=rng)
    raise ValueError(f"Unknown panel algorithm: {algorithm}")


def _worker_soft_panel(
    df: pd.DataFrame,
    attrs: List[AttributeConfig],
    panel_size: int,
    seed: int,
    worker_samples: int,
    algorithm: str,
) -> tuple[pd.Series, int]:
    rng_local = random.Random(seed)
    local_counts = pd.Series(0, index=df.index, dtype=float)
    local_successes = 0
    for _ in range(worker_samples):
        try:
            panel = sample_panel(df, attrs, panel_size, algorithm=algorithm, rng=rng_local)
        except RuntimeError:
            continue
        local_counts.loc[panel.index] += 1
        local_successes += 1
    return local_counts, local_successes


def estimate_selection_probabilities(
    df: pd.DataFrame,
    attrs: List[AttributeConfig],
    panel_size: int,
    num_samples: int = 2000,
    rng_seed: int = 0,
    num_workers: int = 1,
    algorithm: str = "legacy",
) -> pd.Series:
    """Estimate per-rater selection probabilities for a given panel algorithm."""
    counts = pd.Series(0, index=df.index, dtype=float)
    successes = 0

    if algorithm == "leximin":
        categories, people = _build_stratification_inputs(df, attrs, panel_size)
        committees, probabilities = _leximin_distribution(
            copy.deepcopy(categories), copy.deepcopy(people), panel_size
        )
        if not committees:
            raise RuntimeError("LEXIMIN returned no feasible committees.")
        prob_by_id = {person_id: 0.0 for person_id in people.keys()}
        for committee, prob in zip(committees, probabilities):
            for person_id in committee:
                prob_by_id[person_id] += prob
        mapped = df["user_id"].astype(str).map(prob_by_id).fillna(0.0)
        return mapped.astype(float)

    if num_workers <= 1:
        rng = random.Random(rng_seed)
        try:
            from tqdm import tqdm
            iterator = tqdm(range(num_samples), desc="Sampling panels (soft weights)")
        except ImportError:
            iterator = range(num_samples)

        for _ in iterator:
            try:
                panel = sample_panel(df, attrs, panel_size, algorithm=algorithm, rng=rng)
            except RuntimeError:
                continue
            counts.loc[panel.index] += 1
            successes += 1
    else:
        import concurrent.futures
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None  # type: ignore

        samples_per_worker = num_samples // num_workers
        remainder = num_samples % num_workers
        work_items = []
        for i in range(num_workers):
            extra = 1 if i < remainder else 0
            n = samples_per_worker + extra
            if n > 0:
                work_items.append((rng_seed + i, n))

        pbar = tqdm(total=num_samples, desc="Sampling panels (soft weights)") if tqdm else None
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_n = {
                executor.submit(
                    _worker_soft_panel, df, attrs, panel_size, seed, n_samples, algorithm
                ): n_samples
                for seed, n_samples in work_items
            }
            for fut in concurrent.futures.as_completed(future_to_n):
                local_counts, local_successes = fut.result()
                counts = counts.add(local_counts, fill_value=0)
                successes += local_successes
                if pbar:
                    pbar.update(future_to_n[fut])
        if pbar:
            pbar.close()

    if successes == 0:
        raise RuntimeError("No feasible panels found during probability estimation.")
    return counts / successes


def filter_locale(df: pd.DataFrame, locale: Optional[str]) -> pd.DataFrame:
    if locale is None:
        return df
    return df.loc[df["study_locale"] == locale]


def prepare_panel_data(survey_df: pd.DataFrame, config: PanelConfig) -> pd.DataFrame:
    """Filter by locale (if set) and add flattened attributes."""
    filtered = filter_locale(survey_df, config.locale_filter)
    if filtered.empty:
        raise ValueError("No survey rows left after locale filtering; check locale_filter or input data.")
    return add_attribute_columns(filtered, config.attributes)

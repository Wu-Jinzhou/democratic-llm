#!/usr/bin/env python3
"""
Sortition helpers for building demographic panels from PRISM survey data.

Supports:
- Hard panel selection: sample one panel satisfying quota bounds.
- Soft panel weighting: estimate per-rater selection probabilities via Monte Carlo.

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

import random
from dataclasses import dataclass
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
        target = proportion * panel_size
        lower = max(0, int(np.floor(target * (1 - attr.tolerance))))
        upper = min(panel_size, int(np.ceil(target * (1 + attr.tolerance))))
        bounds[category] = (lower, upper)
    return bounds


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


def sample_panel(
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


def _worker_soft_panel(
    df: pd.DataFrame,
    attrs: List[AttributeConfig],
    panel_size: int,
    seed: int,
    worker_samples: int,
) -> tuple[pd.Series, int]:
    rng_local = random.Random(seed)
    local_counts = pd.Series(0, index=df.index, dtype=float)
    local_successes = 0
    for _ in range(worker_samples):
        try:
            panel = sample_panel(df, attrs, panel_size, rng=rng_local)
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
) -> pd.Series:
    """Monte Carlo estimate of per-rater selection probabilities."""
    counts = pd.Series(0, index=df.index, dtype=float)
    successes = 0

    if num_workers <= 1:
        rng = random.Random(rng_seed)
        try:
            from tqdm import tqdm
            iterator = tqdm(range(num_samples), desc="Sampling panels (soft weights)")
        except ImportError:
            iterator = range(num_samples)

        for _ in iterator:
            try:
                panel = sample_panel(df, attrs, panel_size, rng=rng)
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

        # Use ProcessPoolExecutor for CPU-bound sampling
        # Break work into smaller batches to update progress bar frequently
        batch_size = 10
        work_items = []
        remaining = num_samples
        batch_idx = 0
        
        while remaining > 0:
            n = min(batch_size, remaining)
            # Ensure unique seeds for each batch
            work_items.append((rng_seed + batch_idx, n))
            remaining -= n
            batch_idx += 1

        pbar = tqdm(total=num_samples, desc="Sampling panels (soft weights)") if tqdm else None
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_n = {
                executor.submit(
                    _worker_soft_panel, df, attrs, panel_size, seed, n_samples
                ): n_samples
                for seed, n_samples in work_items
            }
            
            for fut in concurrent.futures.as_completed(future_to_n):
                n_samples = future_to_n[fut]
                local_counts, local_successes = fut.result()
                counts = counts.add(local_counts, fill_value=0)
                successes += local_successes
                if pbar:
                    pbar.update(n_samples)
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

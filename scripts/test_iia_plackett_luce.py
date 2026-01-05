#!/usr/bin/env python3
"""
Hausman–McFadden IIA diagnostic for Plackett–Luce models using bootstrap covariance.

Procedure:
1) Fit PL on full set of rankings.
2) For each dropped alternative, fit PL on reduced set.
3) Use ranking-level bootstrap to estimate covariances of full/reduced estimates.
4) Compute Hausman statistic: (b_r - b_f)^T (V_r - V_f)^(-1) (b_r - b_f).

Notes:
- Estimates are aligned by fixing a reference model to 0 (theta_i - theta_ref).
- We use pseudo-inverse if (V_r - V_f) is singular.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from score_rankings import extract_rankings, fit_plackett_luce, load_listwise  # type: ignore

_BOOT_RANKINGS: List[List[str]] | None = None
_BOOT_MODELS: List[str] | None = None
_BOOT_DROP: str | None = None
_BOOT_REFERENCE: str | None = None
_BOOT_REMAINING: List[str] | None = None
_BOOT_MAX_ITER: int = 1000
_BOOT_TOL: float = 1e-6


def drop_rankings(rankings: List[List[str]], drop_model: str) -> List[List[str]]:
    reduced = []
    for r in rankings:
        reduced.append([m for m in r if m != drop_model])
    return reduced


def scores_map(rankings: List[List[str]], models: List[str], max_iter: int, tol: float) -> Dict[str, float]:
    scores, _ = fit_plackett_luce(rankings, models, max_iter=max_iter, tol=tol)
    return {m: float(scores[i]) for i, m in enumerate(models)}


def align_scores(
    score_map: Dict[str, float],
    remaining_models: List[str],
    reference: str,
) -> np.ndarray:
    return np.array([score_map[m] - score_map[reference] for m in remaining_models if m != reference])


def _init_bootstrap(
    rankings: List[List[str]],
    models: List[str],
    drop_model: str,
    reference: str,
    remaining: List[str],
    max_iter: int,
    tol: float,
) -> None:
    global _BOOT_RANKINGS, _BOOT_MODELS, _BOOT_DROP, _BOOT_REFERENCE, _BOOT_REMAINING, _BOOT_MAX_ITER, _BOOT_TOL
    _BOOT_RANKINGS = rankings
    _BOOT_MODELS = models
    _BOOT_DROP = drop_model
    _BOOT_REFERENCE = reference
    _BOOT_REMAINING = remaining
    _BOOT_MAX_ITER = max_iter
    _BOOT_TOL = tol


def _bootstrap_worker(args: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    sample_id, seed, n_rankings = args
    rng = np.random.default_rng(seed + (sample_id * 1000003))
    idx = rng.integers(0, n_rankings, size=n_rankings)
    rankings = _BOOT_RANKINGS
    models = _BOOT_MODELS
    drop_model = _BOOT_DROP
    reference = _BOOT_REFERENCE
    remaining = _BOOT_REMAINING
    if rankings is None or models is None or drop_model is None or reference is None or remaining is None:
        raise RuntimeError("Bootstrap worker not initialized.")
    sample = [rankings[i] for i in idx]
    full_scores = scores_map(sample, models, _BOOT_MAX_ITER, _BOOT_TOL)
    reduced_rankings = drop_rankings(sample, drop_model)
    reduced_models = [m for m in models if m != drop_model]
    reduced_scores = scores_map(reduced_rankings, reduced_models, _BOOT_MAX_ITER, _BOOT_TOL)
    full_vec = align_scores(full_scores, remaining, reference)
    red_vec = align_scores(reduced_scores, remaining, reference)
    return full_vec, red_vec


def hausman_stat(diff: np.ndarray, v_diff: np.ndarray) -> Tuple[float, float, float]:
    eigvals = np.linalg.eigvalsh(v_diff)
    min_eig = float(np.min(eigvals))
    max_eig = float(np.max(eigvals))
    inv = np.linalg.pinv(v_diff)
    stat = float(diff.T @ inv @ diff)
    return stat, min_eig, max_eig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hausman–McFadden IIA test for Plackett–Luce.")
    parser.add_argument(
        "--listwise",
        type=Path,
        default=Path("artifacts/evaluations/listwise.jsonl"),
        help="Listwise rankings JSONL.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluations/iia_plackett_luce.json"))
    parser.add_argument("--drop-models", nargs="*", default=[], help="Models to drop (default: all).")
    parser.add_argument("--reference-model", default=None, help="Reference model for normalization.")
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-workers", type=int, default=1)
    parser.add_argument("--pl-max-iter", type=int, default=1000)
    parser.add_argument("--pl-tol", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_listwise(args.listwise)
    rankings, models, skipped = extract_rankings(records)
    if skipped:
        print(f"Skipped {skipped} rankings with missing/extra models.")
    if not rankings:
        raise RuntimeError("No rankings found.")

    drop_models = args.drop_models or models
    results = []

    for drop in drop_models:
        if drop not in models:
            print(f"Warning: drop model not in data: {drop}")
            continue
        remaining = [m for m in models if m != drop]
        reference = args.reference_model or remaining[0]
        if reference not in remaining:
            reference = remaining[0]

        full_scores = scores_map(rankings, models, args.pl_max_iter, args.pl_tol)
        reduced_rankings = drop_rankings(rankings, drop)
        reduced_models = remaining
        reduced_scores = scores_map(reduced_rankings, reduced_models, args.pl_max_iter, args.pl_tol)

        full_vec = align_scores(full_scores, remaining, reference)
        red_vec = align_scores(reduced_scores, remaining, reference)
        diff = red_vec - full_vec

        full_samples = np.zeros((args.bootstrap_samples, len(remaining) - 1), dtype=float)
        red_samples = np.zeros_like(full_samples)

        if args.bootstrap_samples > 0:
            if args.bootstrap_workers <= 1:
                _init_bootstrap(
                    rankings,
                    models,
                    drop,
                    reference,
                    remaining,
                    args.pl_max_iter,
                    args.pl_tol,
                )
                iterator = range(args.bootstrap_samples)
                iterator = tqdm(iterator, desc=f"Bootstrap drop={drop}", unit="sample")
                for s in iterator:
                    vec_full, vec_red = _bootstrap_worker((s, args.bootstrap_seed, len(rankings)))
                    full_samples[s, :] = vec_full
                    red_samples[s, :] = vec_red
            else:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=args.bootstrap_workers,
                    initializer=_init_bootstrap,
                    initargs=(
                        rankings,
                        models,
                        drop,
                        reference,
                        remaining,
                        args.pl_max_iter,
                        args.pl_tol,
                    ),
                ) as ex:
                    iterator = ex.map(
                        _bootstrap_worker,
                        [(s, args.bootstrap_seed, len(rankings)) for s in range(args.bootstrap_samples)],
                    )
                    iterator = tqdm(iterator, total=args.bootstrap_samples, desc=f"Bootstrap drop={drop}")
                    for s, (vec_full, vec_red) in enumerate(iterator):
                        full_samples[s, :] = vec_full
                        red_samples[s, :] = vec_red

        df = len(diff)
        if args.bootstrap_samples > 1:
            cov_full = np.cov(full_samples, rowvar=False)
            cov_red = np.cov(red_samples, rowvar=False)
            v_diff = cov_red - cov_full
            stat, min_eig, max_eig = hausman_stat(diff, v_diff)
            try:
                from scipy.stats import chi2  # type: ignore

                p_value = float(chi2.sf(stat, df))
            except Exception:
                p_value = None
        else:
            v_diff = np.zeros_like(np.outer(diff, diff))
            stat = None
            min_eig = 0.0
            max_eig = 0.0
            p_value = None

        result = {
            "drop_model": drop,
            "reference_model": reference,
            "n_models_full": len(models),
            "n_models_remaining": len(remaining),
            "df": df,
            "hausman_stat": stat,
            "p_value": p_value,
            "min_eig_v_diff": min_eig,
            "max_eig_v_diff": max_eig,
            "theta_full_aligned": {
                m: full_scores[m] - full_scores[reference] for m in remaining if m != reference
            },
            "theta_reduced_aligned": {
                m: reduced_scores[m] - reduced_scores[reference] for m in remaining if m != reference
            },
            "theta_diff_aligned": {m: float(diff[i]) for i, m in enumerate(m for m in remaining if m != reference)},
            "diff_norm_l2": float(np.linalg.norm(diff)),
            "diff_max_abs": float(np.max(np.abs(diff))),
            "bootstrap": {
                "samples": args.bootstrap_samples,
                "seed": args.bootstrap_seed,
                "workers": args.bootstrap_workers,
                "pl_max_iter": args.pl_max_iter,
                "pl_tol": args.pl_tol,
            },
        }
        results.append(result)

    output = {
        "method": "plackett-luce-iia",
        "models": models,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"Wrote IIA results to {args.output}")


if __name__ == "__main__":
    main()

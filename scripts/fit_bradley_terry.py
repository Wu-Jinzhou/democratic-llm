#!/usr/bin/env python3
"""
Fit a Bradley-Terry model from pairwise preference data.

Expects JSONL with fields:
- model_i, model_j
- wins_i, wins_j

Outputs a JSON list of models with scores (log-ability), win counts, and optional bootstrap CIs.
"""
from __future__ import annotations

import argparse
import json
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

_BT_BOOTSTRAP_UNITS: List[List[Tuple[int, int, float, float]]] | None = None
_BT_BOOTSTRAP_N: int = 0
_BT_BOOTSTRAP_MAX_ITER: int = 1000
_BT_BOOTSTRAP_TOL: float = 1e-6


def load_preferences(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def prepare_records(records: List[dict]) -> Tuple[List[str], List[Tuple[str | None, int, int, float, float]]]:
    models = sorted({rec["model_i"] for rec in records} | {rec["model_j"] for rec in records})
    idx = {m: i for i, m in enumerate(models)}
    prepared = []
    for rec in records:
        qid = rec.get("question_id")
        i = idx[rec["model_i"]]
        j = idx[rec["model_j"]]
        prepared.append((qid, i, j, float(rec.get("wins_i", 0)), float(rec.get("wins_j", 0))))
    return models, prepared


def aggregate_matrix(prepared: List[Tuple[str | None, int, int, float, float]], n: int) -> np.ndarray:
    w = np.zeros((n, n), dtype=float)
    for _, i, j, wins_i, wins_j in prepared:
        w[i, j] += wins_i
        w[j, i] += wins_j
    return w


def fit_bradley_terry_matrix(
    w: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False,
    log_every: int = 50,
):
    n = w.shape[0]
    p = np.ones(n, dtype=float)
    for it in range(max_iter):
        p_old = p.copy()
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if i == j:
                    continue
                nij = w[i, j] + w[j, i]
                if nij > 0:
                    denom += nij / (p[i] + p[j])
            wins = w[i].sum()
            if denom > 0:
                p[i] = wins / denom
        gm = np.exp(np.mean(np.log(p)))
        p = p / gm
        log_p = np.log(np.maximum(p, 1e-12))
        log_old = np.log(np.maximum(p_old, 1e-12))
        delta = float(np.max(np.abs(log_p - log_old)))
        if verbose and (it == 0 or (it + 1) % log_every == 0 or delta < tol):
            print(f"[BT] iter={it + 1} max_delta={delta:.6g}")
        if delta < tol:
            break
    scores = np.log(p)
    return scores, p


def fit_bradley_terry(
    records: List[dict],
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False,
    log_every: int = 50,
):
    models, prepared = prepare_records(records)
    w = aggregate_matrix(prepared, len(models))
    scores, abilities = fit_bradley_terry_matrix(
        w, max_iter=max_iter, tol=tol, verbose=verbose, log_every=log_every
    )
    results = []
    for i, m in enumerate(models):
        results.append({
            "model": m,
            "score": float(scores[i]),
            "ability": float(abilities[i]),
            "wins": float(w[i].sum()),
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results, models, prepared


def bootstrap_scores(
    models: List[str],
    prepared: List[Tuple[str | None, int, int, float, float]],
    num_samples: int,
    seed: int,
    max_iter: int,
    tol: float,
    bootstrap_workers: int = 1,
    verbose: bool = False,
    log_every: int = 50,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(models)
    qids = [qid for qid, *_ in prepared if qid is not None]
    if qids:
        by_qid: Dict[str, List[Tuple[int, int, float, float]]] = {}
        for qid, i, j, wins_i, wins_j in prepared:
            by_qid.setdefault(str(qid), []).append((i, j, wins_i, wins_j))
        units = list(by_qid.values())
    else:
        units = [[(i, j, wins_i, wins_j)] for _, i, j, wins_i, wins_j in prepared]

    scores = np.zeros((num_samples, n), dtype=float)
    if bootstrap_workers <= 1:
        iterator = range(num_samples)
        if verbose:
            iterator = tqdm(iterator, desc="BT bootstrap", unit="sample")
        for s in iterator:
            sample_idx = rng.integers(0, len(units), size=len(units))
            w = np.zeros((n, n), dtype=float)
            for idx in sample_idx:
                for i, j, wins_i, wins_j in units[idx]:
                    w[i, j] += wins_i
                    w[j, i] += wins_j
            sample_scores, _ = fit_bradley_terry_matrix(
                w, max_iter=max_iter, tol=tol, verbose=False, log_every=log_every
            )
            scores[s, :] = sample_scores
    else:
        args = [(s, seed, len(units)) for s in range(num_samples)]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=bootstrap_workers,
            initializer=_init_bt_bootstrap,
            initargs=(units, n, max_iter, tol),
        ) as ex:
            iterator = ex.map(_bt_bootstrap_worker, args)
            if verbose:
                iterator = tqdm(iterator, total=num_samples, desc="BT bootstrap", unit="sample")
            for s, sample_scores in enumerate(iterator):
                scores[s, :] = sample_scores
    return scores


def _init_bt_bootstrap(
    units: List[List[Tuple[int, int, float, float]]],
    n: int,
    max_iter: int,
    tol: float,
) -> None:
    global _BT_BOOTSTRAP_UNITS, _BT_BOOTSTRAP_N, _BT_BOOTSTRAP_MAX_ITER, _BT_BOOTSTRAP_TOL
    _BT_BOOTSTRAP_UNITS = units
    _BT_BOOTSTRAP_N = n
    _BT_BOOTSTRAP_MAX_ITER = max_iter
    _BT_BOOTSTRAP_TOL = tol


def _bt_bootstrap_worker(args: tuple[int, int, int]) -> np.ndarray:
    sample_id, seed, num_units = args
    rng = np.random.default_rng(seed + (sample_id * 1000003))
    units = _BT_BOOTSTRAP_UNITS
    n = _BT_BOOTSTRAP_N
    if units is None:
        raise RuntimeError("Bootstrap worker not initialized.")
    sample_idx = rng.integers(0, num_units, size=num_units)
    w = np.zeros((n, n), dtype=float)
    for idx in sample_idx:
        for i, j, wins_i, wins_j in units[idx]:
            w[i, j] += wins_i
            w[j, i] += wins_j
    sample_scores, _ = fit_bradley_terry_matrix(
        w, max_iter=_BT_BOOTSTRAP_MAX_ITER, tol=_BT_BOOTSTRAP_TOL, verbose=False
    )
    return sample_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a Bradley-Terry model from preference data.")
    parser.add_argument("--preferences", type=Path, required=True, help="JSONL preferences file.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluations/bradley_terry_scores.json"))
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--bootstrap-samples", type=int, default=0, help="Number of bootstrap samples for CIs.")
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap-workers", type=int, default=1, help="Parallel workers for bootstrap.")
    parser.add_argument("--verbose", action="store_true", help="Print fitting progress.")
    parser.add_argument("--log-every", type=int, default=50, help="Log every N iterations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_preferences(args.preferences)
    if args.verbose:
        print(f"[BT] records={len(records)}")
    results, models, prepared = fit_bradley_terry(
        records,
        max_iter=args.max_iter,
        tol=args.tol,
        verbose=args.verbose,
        log_every=args.log_every,
    )
    if args.bootstrap_samples > 0:
        score_samples = bootstrap_scores(
            models=models,
            prepared=prepared,
            num_samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
            max_iter=args.max_iter,
            tol=args.tol,
            bootstrap_workers=args.bootstrap_workers,
            verbose=args.verbose,
            log_every=args.log_every,
        )
        lower = np.quantile(score_samples, args.bootstrap_alpha / 2, axis=0)
        upper = np.quantile(score_samples, 1 - args.bootstrap_alpha / 2, axis=0)
        mean = score_samples.mean(axis=0)
        std = score_samples.std(axis=0)
        model_index = {m: i for i, m in enumerate(models)}
        for rec in results:
            i = model_index[rec["model"]]
            rec["score_mean"] = float(mean[i])
            rec["score_std"] = float(std[i])
            rec["score_ci_lower"] = float(lower[i])
            rec["score_ci_upper"] = float(upper[i])
            rec["ability_ci_lower"] = float(np.exp(lower[i]))
            rec["ability_ci_upper"] = float(np.exp(upper[i]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Wrote scores to {args.output}")
    for rank, rec in enumerate(results, 1):
        print(f"{rank:02d}. {rec['model']}: score={rec['score']:.4f} wins={rec['wins']:.0f}")


if __name__ == "__main__":
    main()

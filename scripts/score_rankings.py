#!/usr/bin/env python3
"""
Score models from listwise rankings:
- Plackett-Luce fitting + bootstrap CIs
- Borda and Copeland scores
- Kemeny ranking (brute force / ILP / heuristic)
- Mallows (Kendall) consensus + dispersion + held-out likelihood
"""
from __future__ import annotations

import argparse
import itertools
import json
import concurrent.futures
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

_PL_BOOTSTRAP_RANKINGS: List[List[str]] | None = None
_PL_BOOTSTRAP_MODELS: List[str] | None = None
_PL_BOOTSTRAP_MAX_ITER: int = 1000
_PL_BOOTSTRAP_TOL: float = 1e-6

_MAL_BOOT_RANKINGS: List[List[str]] | None = None
_MAL_BOOT_MODELS: List[str] | None = None
_MAL_BOOT_KEMENY_METHOD: str = "auto"
_MAL_BOOT_KEMENY_BRUTE_MAX: int = 8
_MAL_BOOT_KEMENY_ILP_MAX: int = 12
_MAL_BOOT_KEMENY_ILP_SECONDS: int = 60


def load_listwise(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_rankings(records: List[dict]) -> Tuple[List[List[str]], List[str], int]:
    all_rankings: List[List[str]] = []
    model_union = set()
    for rec in records:
        rankings = rec.get("rankings") or []
        for ranking in rankings:
            if isinstance(ranking, list):
                model_union.update(ranking)
                all_rankings.append(ranking)
    if not model_union:
        raise ValueError("No rankings found. Ensure you are using listwise output.")

    expected = set(model_union)
    filtered: List[List[str]] = []
    skipped = 0
    for ranking in all_rankings:
        if set(ranking) != expected or len(ranking) != len(expected):
            skipped += 1
            continue
        filtered.append(ranking)
    models = sorted(expected)
    return filtered, models, skipped


def apply_model_drop(
    rankings: List[List[str]],
    models: List[str],
    drop_models: List[str],
) -> Tuple[List[List[str]], List[str], int]:
    if not drop_models:
        return rankings, models, 0
    drop_set = set(drop_models)
    unknown = drop_set - set(models)
    if unknown:
        print(f"Warning: drop models not found in rankings: {sorted(unknown)}")
    drop_set &= set(models)
    remaining = [m for m in models if m not in drop_set]
    expected = set(remaining)
    filtered: List[List[str]] = []
    skipped = 0
    for ranking in rankings:
        reduced = [m for m in ranking if m not in drop_set]
        if set(reduced) != expected or len(reduced) != len(remaining):
            skipped += 1
            continue
        filtered.append(reduced)
    return filtered, sorted(remaining), skipped


def pairwise_counts(rankings: List[List[str]], models: List[str]) -> np.ndarray:
    idx = {m: i for i, m in enumerate(models)}
    n = len(models)
    w = np.zeros((n, n), dtype=float)
    for ranking in rankings:
        for i, winner in enumerate(ranking):
            wi = idx[winner]
            for loser in ranking[i + 1 :]:
                w[wi, idx[loser]] += 1
    return w


def borda_scores(rankings: List[List[str]], models: List[str]) -> List[dict]:
    n = len(models)
    idx = {m: i for i, m in enumerate(models)}
    scores = np.zeros(n, dtype=float)
    for ranking in rankings:
        for pos, model in enumerate(ranking):
            scores[idx[model]] += (n - 1 - pos)
    avg = scores / max(len(rankings), 1)
    results = []
    for i, m in enumerate(models):
        results.append({"model": m, "borda": float(scores[i]), "borda_avg": float(avg[i])})
    results.sort(key=lambda x: x["borda_avg"], reverse=True)
    return results


def copeland_scores(w: np.ndarray, models: List[str]) -> List[dict]:
    n = len(models)
    results = []
    for i, m in enumerate(models):
        wins = float(w[i].sum())
        losses = float(w[:, i].sum())
        total = wins + losses
        win_rate = wins / total if total > 0 else 0.0
        copeland = wins - losses
        results.append(
            {
                "model": m,
                "wins": wins,
                "losses": losses,
                "copeland": copeland,
                "win_rate": win_rate,
            }
        )
    results.sort(key=lambda x: x["copeland"], reverse=True)
    return results


def kendall_distance(ranking: List[str], reference: List[str]) -> int:
    pos_rank = {m: i for i, m in enumerate(ranking)}
    pos_ref = {m: i for i, m in enumerate(reference)}
    distance = 0
    ref_order = sorted(reference, key=lambda m: pos_ref[m])
    for i, a in enumerate(ref_order[:-1]):
        for b in ref_order[i + 1 :]:
            if pos_rank[a] > pos_rank[b]:
                distance += 1
    return distance


def total_kendall_distance(rankings: List[List[str]], reference: List[str]) -> int:
    return sum(kendall_distance(r, reference) for r in rankings)


def mallows_log_z(m: int, phi: float) -> float:
    if phi <= 1e-12:
        return float(math.lgamma(m + 1))
    q = math.exp(-phi)
    if q >= 1.0:
        return float(math.lgamma(m + 1))
    log_num = 0.0
    for i in range(1, m + 1):
        log_num += math.log1p(-q**i)
    log_den = m * math.log1p(-q)
    return log_num - log_den


def mallows_log_likelihood(
    rankings: List[List[str]], reference: List[str], phi: float
) -> float:
    m = len(reference)
    dist_sum = sum(kendall_distance(r, reference) for r in rankings)
    return -phi * dist_sum - len(rankings) * mallows_log_z(m, phi)


def fit_mallows_phi(
    rankings: List[List[str]],
    reference: List[str],
    phi_max: float,
    phi_grid: int,
    phi_tol: float,
) -> Tuple[float, float]:
    if not rankings:
        return 0.0, float("-inf")
    m = len(reference)
    dist_sum = sum(kendall_distance(r, reference) for r in rankings)
    n = len(rankings)

    def ll(phi: float) -> float:
        return -phi * dist_sum - n * mallows_log_z(m, phi)

    grid = np.linspace(0.0, phi_max, max(phi_grid, 2))
    ll_vals = [ll(phi) for phi in grid]
    best_idx = int(np.argmax(ll_vals))
    best_phi = float(grid[best_idx])

    if 0 < best_idx < len(grid) - 1:
        a = float(grid[best_idx - 1])
        b = float(grid[best_idx + 1])
        gr = (math.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        ll_c = ll(c)
        ll_d = ll(d)
        while abs(b - a) > phi_tol:
            if ll_c > ll_d:
                b = d
                d = c
                ll_d = ll_c
                c = b - gr * (b - a)
                ll_c = ll(c)
            else:
                a = c
                c = d
                ll_c = ll_d
                d = a + gr * (b - a)
                ll_d = ll(d)
        best_phi = (a + b) / 2
    return best_phi, ll(best_phi)


def pl_log_likelihood(rankings: List[List[str]], models: List[str], abilities: np.ndarray) -> float:
    idx = {m: i for i, m in enumerate(models)}
    ll = 0.0
    for ranking in rankings:
        ids = [idx[m] for m in ranking]
        for t, i in enumerate(ids):
            denom = abilities[ids[t:]].sum()
            ll += math.log(max(abilities[i], 1e-12)) - math.log(max(denom, 1e-12))
    return ll


def fit_plackett_luce(
    rankings: List[List[str]],
    models: List[str],
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False,
    log_every: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(models)
    idx = {m: i for i, m in enumerate(models)}
    w = np.ones(n, dtype=float)
    for it in range(max_iter):
        w_old = w.copy()
        denom = np.zeros(n, dtype=float)
        count = np.zeros(n, dtype=float)
        for ranking in rankings:
            ids = [idx[m] for m in ranking]
            for t, i in enumerate(ids):
                count[i] += 1.0
                denom_sum = w[ids[t:]].sum()
                denom[i] += 1.0 / max(denom_sum, 1e-12)
        for i in range(n):
            if denom[i] > 0:
                w[i] = count[i] / denom[i]
        gm = np.exp(np.mean(np.log(w)))
        w = w / gm
        log_w = np.log(np.maximum(w, 1e-12))
        log_old = np.log(np.maximum(w_old, 1e-12))
        delta = float(np.max(np.abs(log_w - log_old)))
        if verbose and (it == 0 or (it + 1) % log_every == 0 or delta < tol):
            print(f"[PL] iter={it + 1} max_delta={delta:.6g}")
        if delta < tol:
            break
    return np.log(w), w


def bootstrap_plackett_luce(
    rankings: List[List[str]],
    models: List[str],
    num_samples: int,
    seed: int,
    max_iter: int,
    tol: float,
    bootstrap_workers: int = 1,
    verbose: bool = False,
    log_every: int = 50,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(rankings)
    m = len(models)
    samples = np.zeros((num_samples, m), dtype=float)
    if bootstrap_workers <= 1:
        iterator = range(num_samples)
        if verbose:
            iterator = tqdm(iterator, desc="PL bootstrap", unit="sample")
        for s in iterator:
            sample_idx = rng.integers(0, n, size=n)
            sample_rankings = [rankings[i] for i in sample_idx]
            scores, _ = fit_plackett_luce(
                sample_rankings, models, max_iter=max_iter, tol=tol, verbose=False, log_every=log_every
            )
            samples[s, :] = scores
    else:
        args = [(s, seed, n) for s in range(num_samples)]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=bootstrap_workers,
            initializer=_init_pl_bootstrap,
            initargs=(rankings, models, max_iter, tol),
        ) as ex:
            iterator = ex.map(_pl_bootstrap_worker, args)
            if verbose:
                iterator = tqdm(iterator, total=num_samples, desc="PL bootstrap", unit="sample")
            for s, scores in enumerate(iterator):
                samples[s, :] = scores
    return samples


def _init_pl_bootstrap(
    rankings: List[List[str]],
    models: List[str],
    max_iter: int,
    tol: float,
) -> None:
    global _PL_BOOTSTRAP_RANKINGS, _PL_BOOTSTRAP_MODELS, _PL_BOOTSTRAP_MAX_ITER, _PL_BOOTSTRAP_TOL
    _PL_BOOTSTRAP_RANKINGS = rankings
    _PL_BOOTSTRAP_MODELS = models
    _PL_BOOTSTRAP_MAX_ITER = max_iter
    _PL_BOOTSTRAP_TOL = tol


def _pl_bootstrap_worker(args: tuple[int, int, int]) -> np.ndarray:
    sample_id, seed, n_rankings = args
    rng = np.random.default_rng(seed + (sample_id * 1000003))
    sample_idx = rng.integers(0, n_rankings, size=n_rankings)
    rankings = _PL_BOOTSTRAP_RANKINGS
    models = _PL_BOOTSTRAP_MODELS
    if rankings is None or models is None:
        raise RuntimeError("Bootstrap worker not initialized.")
    sample_rankings = [rankings[i] for i in sample_idx]
    scores, _ = fit_plackett_luce(
        sample_rankings,
        models,
        max_iter=_PL_BOOTSTRAP_MAX_ITER,
        tol=_PL_BOOTSTRAP_TOL,
        verbose=False,
    )
    return scores


def _init_mallows_bootstrap(
    rankings: List[List[str]],
    models: List[str],
    kemeny_method: str,
    kemeny_bruteforce_max: int,
    kemeny_ilp_max: int,
    kemeny_ilp_max_seconds: int,
) -> None:
    global _MAL_BOOT_RANKINGS, _MAL_BOOT_MODELS, _MAL_BOOT_KEMENY_METHOD
    global _MAL_BOOT_KEMENY_BRUTE_MAX, _MAL_BOOT_KEMENY_ILP_MAX, _MAL_BOOT_KEMENY_ILP_SECONDS
    _MAL_BOOT_RANKINGS = rankings
    _MAL_BOOT_MODELS = models
    _MAL_BOOT_KEMENY_METHOD = kemeny_method
    _MAL_BOOT_KEMENY_BRUTE_MAX = kemeny_bruteforce_max
    _MAL_BOOT_KEMENY_ILP_MAX = kemeny_ilp_max
    _MAL_BOOT_KEMENY_ILP_SECONDS = kemeny_ilp_max_seconds


def _mallows_bootstrap_worker(args: Tuple[int, int, int]) -> List[str]:
    sample_id, seed, n_rankings = args
    rankings = _MAL_BOOT_RANKINGS
    models = _MAL_BOOT_MODELS
    if rankings is None or models is None:
        raise RuntimeError("Mallows bootstrap worker not initialized.")
    rng = np.random.default_rng(seed + (sample_id * 1000003))
    sample_idx = rng.integers(0, n_rankings, size=n_rankings)
    sample = [rankings[i] for i in sample_idx]
    consensus_b, _ = compute_consensus(
        sample,
        models,
        _MAL_BOOT_KEMENY_METHOD,
        _MAL_BOOT_KEMENY_BRUTE_MAX,
        _MAL_BOOT_KEMENY_ILP_MAX,
        _MAL_BOOT_KEMENY_ILP_SECONDS,
    )
    return consensus_b


def kemeny_score(order: List[int], w: np.ndarray) -> float:
    score = 0.0
    for i in range(len(order)):
        wi = order[i]
        for j in range(i + 1, len(order)):
            wj = order[j]
            score += w[wi, wj]
    return score


def kemeny_bruteforce(w: np.ndarray) -> Tuple[List[int], float]:
    n = w.shape[0]
    best_order = None
    best_score = -1.0
    for perm in itertools.permutations(range(n)):
        score = kemeny_score(list(perm), w)
        if score > best_score:
            best_score = score
            best_order = list(perm)
    return best_order or list(range(n)), best_score


def kemeny_ilp(w: np.ndarray, max_seconds: int | None = None) -> Tuple[List[int], float]:
    try:
        from mip import BINARY, Model, OptimizationStatus, maximize, xsum
    except Exception as exc:
        raise ImportError("python-mip is required for ILP Kemeny.") from exc

    n = w.shape[0]
    model = Model(sense=maximize, solver_name="CBC")
    model.verbose = 0
    x = [[None for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x[i][j] = model.add_var(var_type=BINARY, name=f"x_{i}_{j}")
    for i in range(n):
        for j in range(i + 1, n):
            model += x[i][j] + x[j][i] == 1
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(n):
                if k in (i, j):
                    continue
                model += x[i][j] + x[j][k] + x[k][i] >= 1
                model += x[i][j] + x[j][k] + x[k][i] <= 2
    model.objective = xsum(w[i, j] * x[i][j] for i in range(n) for j in range(n) if i != j)
    if max_seconds is not None:
        model.max_seconds = max_seconds
    status = model.optimize()
    if status not in {OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE}:
        raise RuntimeError("ILP did not find a feasible solution.")
    wins = []
    for i in range(n):
        ahead = sum(1 for j in range(n) if i != j and x[i][j].x >= 0.5)
        wins.append((i, ahead))
    order = [i for i, _ in sorted(wins, key=lambda t: t[1], reverse=True)]
    score = kemeny_score(order, w)
    return order, score


def kemeny_heuristic(w: np.ndarray, initial: List[int]) -> Tuple[List[int], float]:
    order = initial[:]
    improved = True
    best_score = kemeny_score(order, w)
    while improved:
        improved = False
        for i in range(len(order) - 1):
            candidate = order[:]
            candidate[i], candidate[i + 1] = candidate[i + 1], candidate[i]
            score = kemeny_score(candidate, w)
            if score > best_score:
                order = candidate
                best_score = score
                improved = True
                break
    return order, best_score


def compute_consensus(
    rankings: List[List[str]],
    models: List[str],
    method: str,
    kemeny_bruteforce_max: int,
    kemeny_ilp_max: int,
    kemeny_ilp_max_seconds: int,
) -> Tuple[List[str], str]:
    w = pairwise_counts(rankings, models)
    n = len(models)
    if method == "auto":
        if n <= kemeny_bruteforce_max:
            resolved = "bruteforce"
        elif n <= kemeny_ilp_max:
            resolved = "ilp"
        else:
            resolved = "heuristic"
    else:
        resolved = method
    if resolved == "bruteforce":
        order, _ = kemeny_bruteforce(w)
    elif resolved == "ilp":
        order, _ = kemeny_ilp(w, max_seconds=kemeny_ilp_max_seconds)
    else:
        borda_order = [models.index(rec["model"]) for rec in borda_scores(rankings, models)]
        order, _ = kemeny_heuristic(w, borda_order)
    consensus = [models[i] for i in order]
    return consensus, resolved


def mallows_pairwise_probabilities(
    consensus: List[str],
    phi: float,
    max_enum: int = 8,
    mc_samples: int = 20000,
    seed: int = 42,
) -> Tuple[Dict[str, Dict[str, float]], str]:
    models = list(consensus)
    m = len(models)
    idx = {m: i for i, m in enumerate(models)}
    counts = np.zeros((m, m), dtype=float)

    if m <= max_enum:
        weights = []
        perms = []
        for perm in itertools.permutations(models):
            dist = kendall_distance(list(perm), consensus)
            weight = math.exp(-phi * dist)
            weights.append(weight)
            perms.append(perm)
        total = sum(weights)
        for perm, weight in zip(perms, weights):
            pos = {m: i for i, m in enumerate(perm)}
            for i in range(m):
                for j in range(m):
                    if i == j:
                        continue
                    if pos[models[i]] < pos[models[j]]:
                        counts[i, j] += weight
        probs = counts / max(total, 1e-12)
        method = "exact"
    else:
        rng = random.Random(seed)
        for _ in range(mc_samples):
            # simple insertion model sampling for Kendall Mallows
            perm = [consensus[0]]
            for t in range(1, m):
                # insertion position distribution proportional to exp(-phi * k)
                weights = [math.exp(-phi * k) for k in range(t + 1)]
                total = sum(weights)
                r = rng.random() * total
                c = 0.0
                pos = 0
                for k, w in enumerate(weights):
                    c += w
                    if r <= c:
                        pos = k
                        break
                perm.insert(pos, consensus[t])
            pos_map = {m: i for i, m in enumerate(perm)}
            for i in range(m):
                for j in range(m):
                    if i == j:
                        continue
                    if pos_map[models[i]] < pos_map[models[j]]:
                        counts[i, j] += 1.0
        probs = counts / max(mc_samples, 1.0)
        method = "mc"

    out: Dict[str, Dict[str, float]] = {}
    for i, mi in enumerate(models):
        out[mi] = {}
        for j, mj in enumerate(models):
            if i == j:
                continue
            out[mi][mj] = float(probs[i, j])
    return out, method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score models from listwise rankings.")
    parser.add_argument(
        "--listwise",
        type=Path,
        default=Path("artifacts/evaluations/listwise.jsonl"),
        help="Listwise evaluation JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluations/ranking_scores.json"),
        help="Output JSON summary.",
    )
    parser.add_argument(
        "--method",
        choices=["plackett-luce", "borda", "copeland", "kemeny", "mallows"],
        default="plackett-luce",
        help="Which scoring method to run.",
    )
    parser.add_argument(
        "--drop-models",
        nargs="*",
        default=[],
        help="Model IDs to drop before scoring (useful for IIA diagnostics).",
    )
    parser.add_argument("--pl-max-iter", type=int, default=1000)
    parser.add_argument("--pl-tol", type=float, default=1e-6)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap-workers", type=int, default=1, help="Parallel workers for bootstrap.")
    parser.add_argument("--kemeny-method", choices=["auto", "bruteforce", "ilp", "heuristic"], default="auto")
    parser.add_argument("--kemeny-bruteforce-max", type=int, default=8)
    parser.add_argument("--kemeny-ilp-max", type=int, default=12)
    parser.add_argument("--kemeny-ilp-max-seconds", type=int, default=60)
    parser.add_argument("--mallows-test-fraction", type=float, default=0.2)
    parser.add_argument("--mallows-seed", type=int, default=42)
    parser.add_argument("--mallows-phi-max", type=float, default=10.0)
    parser.add_argument("--mallows-phi-grid", type=int, default=200)
    parser.add_argument("--mallows-phi-tol", type=float, default=1e-4)
    parser.add_argument("--mallows-bootstrap-samples", type=int, default=0)
    parser.add_argument("--mallows-bootstrap-seed", type=int, default=42)
    parser.add_argument("--mallows-bootstrap-workers", type=int, default=1)
    parser.add_argument("--mallows-pairwise-max-enum", type=int, default=8)
    parser.add_argument("--mallows-pairwise-mc-samples", type=int, default=20000)
    parser.add_argument("--verbose", action="store_true", help="Print scoring progress.")
    parser.add_argument("--log-every", type=int, default=50, help="Log every N iterations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_listwise(args.listwise)
    rankings, models, skipped = extract_rankings(records)
    if args.verbose:
        print(f"[Rankings] total_records={len(records)}")
        print(f"[Rankings] valid_rankings={len(rankings)} skipped={skipped}")
    elif skipped:
        print(f"Skipped {skipped} rankings with missing/extra models.")

    drop_skipped = 0
    if args.drop_models:
        rankings, models, drop_skipped = apply_model_drop(rankings, models, args.drop_models)
        if args.verbose:
            print(f"[Rankings] drop_models={args.drop_models} remaining={len(models)}")
            print(f"[Rankings] dropped_skipped={drop_skipped} remaining_rankings={len(rankings)}")
        elif drop_skipped:
            print(f"Skipped {drop_skipped} rankings after dropping models.")
    if not rankings:
        raise RuntimeError("No valid rankings found.")

    output = {
        "method": args.method,
        "models": models,
        "num_rankings": len(rankings),
        "drop_models": args.drop_models,
        "drop_skipped": drop_skipped,
    }

    if args.method == "plackett-luce":
        pl_scores, pl_abilities = fit_plackett_luce(
            rankings,
            models,
            max_iter=args.pl_max_iter,
            tol=args.pl_tol,
            verbose=args.verbose,
            log_every=args.log_every,
        )
        pl_results = []
        for i, m in enumerate(models):
            pl_results.append(
                {
                    "model": m,
                    "score": float(pl_scores[i]),
                    "ability": float(pl_abilities[i]),
                }
            )
        pl_results.sort(key=lambda x: x["score"], reverse=True)
        if args.bootstrap_samples > 0:
            samples = bootstrap_plackett_luce(
                rankings,
                models,
                num_samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
                max_iter=args.pl_max_iter,
                tol=args.pl_tol,
                bootstrap_workers=args.bootstrap_workers,
                verbose=args.verbose,
                log_every=args.log_every,
            )
            lower = np.quantile(samples, args.bootstrap_alpha / 2, axis=0)
            upper = np.quantile(samples, 1 - args.bootstrap_alpha / 2, axis=0)
            mean = samples.mean(axis=0)
            std = samples.std(axis=0)
            model_index = {m: i for i, m in enumerate(models)}
            for rec in pl_results:
                i = model_index[rec["model"]]
                rec["score_mean"] = float(mean[i])
                rec["score_std"] = float(std[i])
                rec["score_ci_lower"] = float(lower[i])
                rec["score_ci_upper"] = float(upper[i])
                rec["ability_ci_lower"] = float(np.exp(lower[i]))
                rec["ability_ci_upper"] = float(np.exp(upper[i]))
        output["plackett_luce"] = {
            "results": pl_results,
            "max_iter": args.pl_max_iter,
            "tol": args.pl_tol,
            "bootstrap_samples": args.bootstrap_samples,
        }
    elif args.method == "borda":
        if args.verbose:
            print("[Borda] scoring rankings")
        output["borda"] = borda_scores(rankings, models)
    elif args.method == "copeland":
        if args.verbose:
            print("[Copeland] scoring rankings")
        w = pairwise_counts(rankings, models)
        output["copeland"] = copeland_scores(w, models)
    elif args.method == "kemeny":
        w = pairwise_counts(rankings, models)
        n = len(models)
        if args.kemeny_method == "auto":
            if n <= args.kemeny_bruteforce_max:
                method = "bruteforce"
            elif n <= args.kemeny_ilp_max:
                method = "ilp"
            else:
                method = "heuristic"
        else:
            method = args.kemeny_method
        if args.verbose:
            print(f"[Kemeny] method={method} models={n}")
        if method == "bruteforce":
            order, score = kemeny_bruteforce(w)
        elif method == "ilp":
            order, score = kemeny_ilp(w, max_seconds=args.kemeny_ilp_max_seconds)
        else:
            borda_order = [models.index(rec["model"]) for rec in borda_scores(rankings, models)]
            order, score = kemeny_heuristic(w, borda_order)
        output["kemeny"] = {
            "method": method,
            "ranking": [models[i] for i in order],
            "score": float(score),
        }
    elif args.method == "mallows":
        rng = random.Random(args.mallows_seed)
        indices = list(range(len(rankings)))
        rng.shuffle(indices)
        test_size = int(round(len(indices) * args.mallows_test_fraction))
        test_idx = set(indices[:test_size])
        train_rankings = [rankings[i] for i in indices if i not in test_idx]
        test_rankings = [rankings[i] for i in indices if i in test_idx]
        if args.verbose:
            print(f"[Mallows] train={len(train_rankings)} test={len(test_rankings)}")

        consensus, consensus_method = compute_consensus(
            train_rankings,
            models,
            args.kemeny_method,
            args.kemeny_bruteforce_max,
            args.kemeny_ilp_max,
            args.kemeny_ilp_max_seconds,
        )

        phi, ll_train = fit_mallows_phi(
            train_rankings,
            consensus,
            phi_max=args.mallows_phi_max,
            phi_grid=args.mallows_phi_grid,
            phi_tol=args.mallows_phi_tol,
        )
        ll_test = mallows_log_likelihood(test_rankings, consensus, phi) if test_rankings else float("nan")

        bootstrap_results = None
        if args.mallows_bootstrap_samples > 0:
            if args.verbose:
                print(f"[Mallows] bootstrap samples={args.mallows_bootstrap_samples}")
            counts = {m: np.zeros(len(models), dtype=float) for m in models}
            exact_matches = 0
            if args.mallows_bootstrap_workers <= 1:
                _init_mallows_bootstrap(
                    train_rankings,
                    models,
                    args.kemeny_method,
                    args.kemeny_bruteforce_max,
                    args.kemeny_ilp_max,
                    args.kemeny_ilp_max_seconds,
                )
                iterator = range(args.mallows_bootstrap_samples)
                if args.verbose:
                    iterator = tqdm(iterator, desc="Mallows bootstrap", unit="sample")
                for s in iterator:
                    cons = _mallows_bootstrap_worker((s, args.mallows_bootstrap_seed, len(train_rankings)))
                    if cons == consensus:
                        exact_matches += 1
                    for pos, m in enumerate(cons):
                        counts[m][pos] += 1
            else:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=args.mallows_bootstrap_workers,
                    initializer=_init_mallows_bootstrap,
                    initargs=(
                        train_rankings,
                        models,
                        args.kemeny_method,
                        args.kemeny_bruteforce_max,
                        args.kemeny_ilp_max,
                        args.kemeny_ilp_max_seconds,
                    ),
                ) as ex:
                    iterator = ex.map(
                        _mallows_bootstrap_worker,
                        [
                            (s, args.mallows_bootstrap_seed, len(train_rankings))
                            for s in range(args.mallows_bootstrap_samples)
                        ],
                    )
                    if args.verbose:
                        iterator = tqdm(
                            iterator,
                            total=args.mallows_bootstrap_samples,
                            desc="Mallows bootstrap",
                            unit="sample",
                        )
                    for cons in iterator:
                        if cons == consensus:
                            exact_matches += 1
                        for pos, m in enumerate(cons):
                            counts[m][pos] += 1

            rank_probabilities = {
                m: [float(counts[m][i] / args.mallows_bootstrap_samples) for i in range(len(models))]
                for m in models
            }
            bootstrap_results = {
                "samples": args.mallows_bootstrap_samples,
                "seed": args.mallows_bootstrap_seed,
                "workers": args.mallows_bootstrap_workers,
                "exact_match_rate": exact_matches / max(args.mallows_bootstrap_samples, 1),
                "rank_probabilities": rank_probabilities,
            }

        pairwise_probs, pairwise_method = mallows_pairwise_probabilities(
            consensus,
            phi,
            max_enum=args.mallows_pairwise_max_enum,
            mc_samples=args.mallows_pairwise_mc_samples,
            seed=args.mallows_seed,
        )

        output["mallows"] = {
            "consensus": consensus,
            "consensus_method": consensus_method,
            "phi": float(phi),
            "log_likelihood_train": float(ll_train),
            "log_likelihood_test": float(ll_test),
            "params": {
                "test_fraction": args.mallows_test_fraction,
                "seed": args.mallows_seed,
                "phi_max": args.mallows_phi_max,
                "phi_grid": args.mallows_phi_grid,
                "phi_tol": args.mallows_phi_tol,
                "kemeny_method": consensus_method,
                "kemeny_bruteforce_max": args.kemeny_bruteforce_max,
                "kemeny_ilp_max": args.kemeny_ilp_max,
                "kemeny_ilp_max_seconds": args.kemeny_ilp_max_seconds,
                "bootstrap_samples": args.mallows_bootstrap_samples,
                "bootstrap_workers": args.mallows_bootstrap_workers,
            },
            "mean_kendall_train": float(
                total_kendall_distance(train_rankings, consensus) / max(len(train_rankings), 1)
            ),
            "mean_kendall_test": float(
                total_kendall_distance(test_rankings, consensus) / max(len(test_rankings), 1)
            )
            if test_rankings
            else float("nan"),
            "n_train": len(train_rankings),
            "n_test": len(test_rankings),
            "bootstrap": bootstrap_results,
            "pairwise_probabilities": {
                "method": pairwise_method,
                "matrix": pairwise_probs,
            },
        }

        pl_scores, pl_abilities = fit_plackett_luce(
            train_rankings,
            models,
            max_iter=args.pl_max_iter,
            tol=args.pl_tol,
            verbose=args.verbose,
            log_every=args.log_every,
        )
        pl_train_ll = pl_log_likelihood(train_rankings, models, pl_abilities)
        pl_test_ll = pl_log_likelihood(test_rankings, models, pl_abilities) if test_rankings else float("nan")
        pl_results = []
        for i, m in enumerate(models):
            pl_results.append({"model": m, "score": float(pl_scores[i]), "ability": float(pl_abilities[i])})
        pl_results.sort(key=lambda x: x["score"], reverse=True)
        output["plackett_luce_compare"] = {
            "log_likelihood_train": float(pl_train_ll),
            "log_likelihood_test": float(pl_test_ll),
            "results": pl_results,
            "max_iter": args.pl_max_iter,
            "tol": args.pl_tol,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"Wrote scores to {args.output}")


if __name__ == "__main__":
    main()

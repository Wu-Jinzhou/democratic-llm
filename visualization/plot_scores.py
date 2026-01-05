#!/usr/bin/env python3
"""
Plot preference scores from ranking/score JSON files.
Supports: bradley-terry, plackett-luce, borda, copeland, kemeny, mallows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from style import apply_style, style_axes


def _infer_method(data: object) -> str:
    if isinstance(data, list):
        return "bradley-terry"
    if not isinstance(data, dict):
        raise ValueError("Unsupported input format.")
    if "method" in data:
        return str(data["method"])
    if "plackett_luce" in data:
        return "plackett-luce"
    if "borda" in data:
        return "borda"
    if "copeland" in data:
        return "copeland"
    if "kemeny" in data:
        return "kemeny"
    if "mallows" in data:
        return "mallows"
    raise ValueError("Cannot infer method from input.")


def _extract_results(data: object, method: str) -> Tuple[List[dict], List[str], List[float], List[Tuple[float, float]]]:
    if method == "bradley-terry":
        if isinstance(data, list):
            results = data
        elif isinstance(data, dict) and "results" in data:
            results = data["results"]
        else:
            raise ValueError("Bradley-Terry results not found.")
        metric = "score" if "score" in results[0] else "ability"
        models = [r["model"] for r in results]
        values = [float(r.get(metric, 0.0)) for r in results]
        ci_low_key = f"{metric}_ci_lower"
        ci_high_key = f"{metric}_ci_upper"
        intervals = [
            (float(r[ci_low_key]), float(r[ci_high_key]))
            if ci_low_key in r and ci_high_key in r
            else (val, val)
            for r, val in zip(results, values)
        ]
        return results, models, values, intervals

    if not isinstance(data, dict):
        raise ValueError("Ranking scores file is expected to be a JSON object.")

    if method == "plackett-luce":
        results = data.get("plackett_luce", {}).get("results")
        if not isinstance(results, list):
            raise ValueError("Plackett-Luce results not found.")
        metric = "score" if "score" in results[0] else "ability"
        models = [r["model"] for r in results]
        values = [float(r.get(metric, 0.0)) for r in results]
        ci_low_key = f"{metric}_ci_lower"
        ci_high_key = f"{metric}_ci_upper"
        intervals = [
            (float(r[ci_low_key]), float(r[ci_high_key]))
            if ci_low_key in r and ci_high_key in r
            else (val, val)
            for r, val in zip(results, values)
        ]
        return results, models, values, intervals

    if method == "borda":
        results = data.get("borda")
        if not isinstance(results, list):
            raise ValueError("Borda results not found.")
        metric = "borda_avg" if "borda_avg" in results[0] else "borda"
        models = [r["model"] for r in results]
        values = [float(r.get(metric, 0.0)) for r in results]
        intervals = [(val, val) for val in values]
        return results, models, values, intervals

    if method == "copeland":
        results = data.get("copeland")
        if not isinstance(results, list):
            raise ValueError("Copeland results not found.")
        metric = "copeland" if "copeland" in results[0] else "win_rate"
        models = [r["model"] for r in results]
        values = [float(r.get(metric, 0.0)) for r in results]
        intervals = [(val, val) for val in values]
        return results, models, values, intervals

    if method == "kemeny":
        kemeny = data.get("kemeny", {})
        ranking = kemeny.get("ranking")
        if not isinstance(ranking, list):
            raise ValueError("Kemeny ranking not found.")
        models = [str(m) for m in ranking]
        n = len(models)
        values = [float(n - i) for i in range(n)]
        intervals = [(val, val) for val in values]
        results = [{"model": m, "rank": i + 1} for i, m in enumerate(models)]
        return results, models, values, intervals

    if method == "mallows":
        mallows = data.get("mallows", {})
        ranking = mallows.get("consensus")
        if not isinstance(ranking, list):
            raise ValueError("Mallows consensus ranking not found.")
        models = [str(m) for m in ranking]
        n = len(models)
        values = [float(n - i) for i in range(n)]
        intervals = [(val, val) for val in values]
        results = [{"model": m, "rank": i + 1} for i, m in enumerate(models)]
        return results, models, values, intervals

    raise ValueError(f"Unsupported method: {method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot preference scores from ranking/score files.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/evaluations/ranking_scores.json"),
        help="Ranking/score JSON (ranking_scores.json or bradley_terry_scores.json).",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "bradley-terry", "plackett-luce", "borda", "copeland", "kemeny", "mallows"],
        default="auto",
        help="Which method to plot (auto tries to infer).",
    )
    parser.add_argument("--output", type=Path, default=Path("visualization/output/preferences_scores.png"))
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_style()
    data = json.loads(args.input.read_text())
    method = _infer_method(data) if args.method == "auto" else args.method
    _, models, values, intervals = _extract_results(data, method)
    err_low = [v - lo for v, (lo, _) in zip(values, intervals)]
    err_high = [hi - v for v, (_, hi) in zip(values, intervals)]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(models))))
    y = list(range(len(models)))
    if any(err_low) or any(err_high):
        ax.barh(y, values, xerr=[err_low, err_high], capsize=4)
    else:
        ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    if method == "kemeny":
        xlabel = "Kemeny rank score (higher = better)"
    elif method == "mallows":
        xlabel = "Mallows consensus rank score (higher = better)"
    elif method in {"borda"}:
        xlabel = "Borda score"
    elif method in {"copeland"}:
        xlabel = "Copeland score"
    elif method in {"bradley-terry", "plackett-luce"}:
        xlabel = "Score (log-ability)" if any(err_low) or any(err_high) else "Score"
    else:
        xlabel = "Score"
    ax.set_xlabel(xlabel)
    if method == "mallows" and isinstance(data, dict):
        mallows = data.get("mallows", {})
        phi = mallows.get("phi")
        ll_test = mallows.get("log_likelihood_test")
        details = []
        if isinstance(phi, (int, float)):
            details.append(f"phi={phi:.3f}")
        if isinstance(ll_test, (int, float)):
            details.append(f"ll_test={ll_test:.2f}")
        suffix = f" ({', '.join(details)})" if details else ""
        title = args.title or f"Mallows consensus{suffix}"
    else:
        title = args.title or f"{method.replace('-', ' ').title()} scores"
    ax.set_title(title)
    style_axes(ax)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"Wrote plot to {args.output}")


if __name__ == "__main__":
    main()

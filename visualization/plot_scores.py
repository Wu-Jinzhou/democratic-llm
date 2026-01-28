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
import matplotlib.colors as mcolors

from style import apply_style, style_axes, display_model_names, truncated_cmap


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


def _bootstrap_samples(data: object, method: str) -> int | None:
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("bootstrap_samples"), int):
        n = int(data["bootstrap_samples"])
        return n if n > 0 else None
    bootstrap = data.get("bootstrap")
    if isinstance(bootstrap, dict) and isinstance(bootstrap.get("samples"), int):
        n = int(bootstrap["samples"])
        return n if n > 0 else None
    if method == "plackett-luce":
        pl = data.get("plackett_luce", {})
        if isinstance(pl, dict):
            if isinstance(pl.get("bootstrap_samples"), int):
                n = int(pl["bootstrap_samples"])
                return n if n > 0 else None
            pl_boot = pl.get("bootstrap")
            if isinstance(pl_boot, dict) and isinstance(pl_boot.get("samples"), int):
                n = int(pl_boot["samples"])
                return n if n > 0 else None
    if method == "mallows":
        mallows = data.get("mallows", {})
        if isinstance(mallows, dict):
            m_boot = mallows.get("bootstrap")
            if isinstance(m_boot, dict) and isinstance(m_boot.get("samples"), int):
                n = int(m_boot["samples"])
                return n if n > 0 else None
    if method == "kemeny":
        kemeny = data.get("kemeny", {})
        if isinstance(kemeny, dict):
            k_boot = kemeny.get("bootstrap")
            if isinstance(k_boot, dict) and isinstance(k_boot.get("samples"), int):
                n = int(k_boot["samples"])
                return n if n > 0 else None
    return None


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
        ci_low_key = f"{metric}_ci_lower"
        ci_high_key = f"{metric}_ci_upper"
        intervals = [
            (float(r[ci_low_key]), float(r[ci_high_key]))
            if ci_low_key in r and ci_high_key in r
            else (val, val)
            for r, val in zip(results, values)
        ]
        return results, models, values, intervals

    if method == "copeland":
        results = data.get("copeland")
        if not isinstance(results, list):
            raise ValueError("Copeland results not found.")
        metric = "copeland" if "copeland" in results[0] else "win_rate"
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


def _draw_rank_heatmap(
    ax,
    matrix,
    row_models: List[str],
    title: str,
    bootstrap_n: int | None = None,
) -> None:
    ax.set_facecolor("white")
    cmap = truncated_cmap("Blues", minval=0.22, maxval=0.97)
    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")

    n_models = len(row_models)
    ax.set_xticks(list(range(n_models)))
    ax.set_xticklabels([str(i + 1) for i in range(n_models)])
    ax.set_yticks(list(range(n_models)))
    ax.set_yticklabels(display_model_names(row_models))
    ax.set_xlabel("Rank position")
    ax.set_ylabel("")

    # Light separators between cells (no background grid).
    ax.set_xticks([x - 0.5 for x in range(1, n_models)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, n_models)], minor=True)
    ax.grid(False)
    ax.grid(which="minor", color="#FFFFFF", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title(title)

    # Apply shared title styling (Roboto + bold) and consistent ticks.
    style_axes(ax, grid=False)

    # Hide spines for a cleaner heatmap panel (after style_axes, which tweaks spines).
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Annotate probabilities inside each cell (7x7 is small enough to be readable).
    n_models = len(row_models)
    for i in range(n_models):
        for j in range(n_models):
            val = float(matrix[i][j])
            color = "white" if val >= 0.5 else "#222222"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )


def main() -> None:
    args = parse_args()
    apply_style(grid=False)
    data = json.loads(args.input.read_text())
    method = _infer_method(data) if args.method == "auto" else args.method
    _, models, values, intervals = _extract_results(data, method)
    err_low = [v - lo for v, (lo, _) in zip(values, intervals)]
    err_high = [hi - v for v, (_, hi) in zip(values, intervals)]
    # Bradley–Terry outputs may be stored as a bare list (legacy format). In that case we
    # cannot recover bootstrap sample count from the file, but if error bars are present
    # we still label the plot as bootstrapped for clarity.
    inferred_bt_bootstrap_n: int | None = None
    if method == "bradley-terry" and (any(err_low) or any(err_high)):
        inferred_bt_bootstrap_n = 100

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(models))))
    heatmap_mode = False
    if method in {"kemeny", "mallows"} and isinstance(data, dict):
        heatmap_mode = True
        if method == "kemeny":
            kemeny = data.get("kemeny", {})
            ranking = kemeny.get("ranking")
            if not isinstance(ranking, list):
                raise ValueError("Kemeny ranking not found.")
            row_models = [str(m) for m in ranking]
            n = len(row_models)
            probs = kemeny.get("bootstrap", {}).get("rank_probabilities")
            if isinstance(probs, dict):
                matrix = []
                for m in row_models:
                    row = probs.get(m)
                    if not isinstance(row, list) or len(row) != n:
                        raise ValueError(f"Missing/invalid rank probabilities for model: {m}")
                    matrix.append([float(x) for x in row])
            else:
                matrix = [[0.0] * n for _ in range(n)]
                for i in range(n):
                    matrix[i][i] = 1.0
            _draw_rank_heatmap(
                ax,
                matrix,
                row_models=row_models,
                title=args.title or "Kemeny rank-position probabilities",
                bootstrap_n=_bootstrap_samples(data, method),
            )
        else:
            mallows = data.get("mallows", {})
            row_models = mallows.get("consensus")
            if not isinstance(row_models, list):
                raise ValueError("Mallows consensus ranking not found.")
            row_models = [str(m) for m in row_models]
            n = len(row_models)
            probs = mallows.get("bootstrap", {}).get("rank_probabilities")
            if isinstance(probs, dict):
                matrix = []
                for m in row_models:
                    row = probs.get(m)
                    if not isinstance(row, list) or len(row) != n:
                        raise ValueError(f"Missing/invalid rank probabilities for model: {m}")
                    matrix.append([float(x) for x in row])
            else:
                # Fallback: deterministic one-hot from the consensus ranking.
                matrix = [[0.0] * n for _ in range(n)]
                for i in range(n):
                    matrix[i][i] = 1.0

            phi = mallows.get("phi")
            ll_test = mallows.get("log_likelihood_test")
            details = []
            if isinstance(phi, (int, float)):
                details.append(f"phi={phi:.3f}")
            if isinstance(ll_test, (int, float)):
                details.append(f"ll_test={ll_test:.2f}")
            detail_text = f" ({', '.join(details)})" if details else ""
            _draw_rank_heatmap(
                ax,
                matrix,
                row_models=row_models,
                title=args.title or f"Mallows rank-position probabilities{detail_text}",
                bootstrap_n=_bootstrap_samples(data, method),
            )
    else:
        y = list(range(len(models)))
        vmin = min(values) if values else 0.0
        vmax = max(values) if values else 1.0
        if vmin < 0 < vmax:
            norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
            cmap = truncated_cmap("Blues", minval=0.35, maxval=0.95)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = truncated_cmap("Blues", minval=0.35, maxval=0.95)
        colors = cmap(norm(values))
        if any(err_low) or any(err_high):
            ax.barh(y, values, xerr=[err_low, err_high], capsize=4, color=colors, edgecolor="none")
        else:
            ax.barh(y, values, color=colors, edgecolor="none")
        ax.set_yticks(y)
        ax.set_yticklabels(display_model_names(models))
        ax.invert_yaxis()
        if method in {"borda"}:
            xlabel = "Borda score"
        elif method in {"copeland"}:
            xlabel = "Copeland score"
        elif method in {"bradley-terry", "plackett-luce"}:
            xlabel = "Score (log-ability)" if any(err_low) or any(err_high) else "Score"
        else:
            xlabel = "Score"
        ax.set_xlabel(xlabel)
    if heatmap_mode:
        title = None
    elif method == "mallows" and isinstance(data, dict):
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
    if title:
        ax.set_title(title)
        style_axes(ax, grid=False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"Wrote plot to {args.output}")


if __name__ == "__main__":
    main()

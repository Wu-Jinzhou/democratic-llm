#!/usr/bin/env python3
"""
Compare bounded gradient diagnostics between two training runs.

Expected inputs per run directory:
- metadata.json
- gradient_summaries.csv
- gradient_sketches.npz
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare hard-vs-soft gradient diagnostics.")
    parser.add_argument("--hard-dir", type=Path, required=True)
    parser.add_argument("--soft-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_summary_csv(path: Path) -> dict[str, list[dict[str, float | int | str]]]:
    rows_by_layer: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {
                "step": int(row["step"]),
                "layer": row["layer"],
                "numel": int(row["numel"]),
                "l1": float(row["l1"]),
                "l2": float(row["l2"]),
                "l1_over_l2": float(row["l1_over_l2"]),
                "l1_over_sqrt_d_l2": float(row["l1_over_sqrt_d_l2"]),
            }
            rows_by_layer[row["layer"]].append(parsed)
    return rows_by_layer


def load_sketches(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["steps"], data["coords"], data["vectors"]


def summarize_metric(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
    }


def bootstrap_mean_diff(hard: np.ndarray, soft: np.ndarray, samples: int, seed: int) -> dict[str, float]:
    if hard.size == 0 or soft.size == 0:
        return {"delta_mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(samples):
        hard_sample = rng.choice(hard, size=hard.size, replace=True)
        soft_sample = rng.choice(soft, size=soft.size, replace=True)
        deltas.append(float(np.mean(hard_sample) - np.mean(soft_sample)))
    deltas_arr = np.asarray(deltas, dtype=np.float64)
    return {
        "delta_mean": float(np.mean(hard) - np.mean(soft)),
        "ci_low": float(np.quantile(deltas_arr, 0.025)),
        "ci_high": float(np.quantile(deltas_arr, 0.975)),
    }


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def cosine_matrix(matrix: np.ndarray) -> np.ndarray:
    normalized = normalize_rows(matrix)
    if normalized.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    cosine = normalized @ normalized.T
    return np.clip(cosine, -1.0, 1.0)


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.zeros((0,), dtype=np.float32)
    tri = np.triu_indices(matrix.shape[0], k=1)
    return matrix[tri]


def effective_rank(matrix: np.ndarray) -> dict[str, float]:
    if matrix.size == 0 or matrix.shape[0] == 0:
        return {"effective_rank": float("nan"), "stable_rank": float("nan")}
    normalized = normalize_rows(matrix)
    singular_values = np.linalg.svd(normalized, full_matrices=False, compute_uv=False)
    if singular_values.size == 0 or float(np.sum(singular_values)) == 0.0:
        return {"effective_rank": 0.0, "stable_rank": 0.0}
    probs = singular_values / np.sum(singular_values)
    entropy = -np.sum(np.where(probs > 0, probs * np.log(probs), 0.0))
    stable = float(np.sum(singular_values**2) / max(float(singular_values[0] ** 2), 1e-12))
    return {"effective_rank": float(np.exp(entropy)), "stable_rank": stable}


def per_layer_table(
    hard_rows: dict[str, list[dict[str, float | int | str]]],
    soft_rows: dict[str, list[dict[str, float | int | str]]],
) -> list[dict[str, float | int | str]]:
    layers = sorted(set(hard_rows.keys()) | set(soft_rows.keys()))
    out_rows = []
    for layer in layers:
        hard_metric = np.asarray([float(r["l1_over_l2"]) for r in hard_rows.get(layer, [])], dtype=np.float64)
        soft_metric = np.asarray([float(r["l1_over_l2"]) for r in soft_rows.get(layer, [])], dtype=np.float64)
        hard_norm = np.asarray([float(r["l1_over_sqrt_d_l2"]) for r in hard_rows.get(layer, [])], dtype=np.float64)
        soft_norm = np.asarray([float(r["l1_over_sqrt_d_l2"]) for r in soft_rows.get(layer, [])], dtype=np.float64)
        out_rows.append(
            {
                "layer": layer,
                "hard_steps": int(hard_metric.size),
                "soft_steps": int(soft_metric.size),
                "hard_l1_over_l2_mean": float(np.mean(hard_metric)) if hard_metric.size else float("nan"),
                "soft_l1_over_l2_mean": float(np.mean(soft_metric)) if soft_metric.size else float("nan"),
                "delta_l1_over_l2_mean": float(np.mean(hard_metric) - np.mean(soft_metric))
                if hard_metric.size and soft_metric.size
                else float("nan"),
                "hard_l1_over_l2_median": float(np.median(hard_metric)) if hard_metric.size else float("nan"),
                "soft_l1_over_l2_median": float(np.median(soft_metric)) if soft_metric.size else float("nan"),
                "hard_l1_over_sqrt_d_l2_mean": float(np.mean(hard_norm)) if hard_norm.size else float("nan"),
                "soft_l1_over_sqrt_d_l2_mean": float(np.mean(soft_norm)) if soft_norm.size else float("nan"),
                "delta_l1_over_sqrt_d_l2_mean": float(np.mean(hard_norm) - np.mean(soft_norm))
                if hard_norm.size and soft_norm.size
                else float("nan"),
                "hard_l1_over_sqrt_d_l2_median": float(np.median(hard_norm)) if hard_norm.size else float("nan"),
                "soft_l1_over_sqrt_d_l2_median": float(np.median(soft_norm)) if soft_norm.size else float("nan"),
            }
        )
    return out_rows


def write_per_layer_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hard_meta = load_metadata(args.hard_dir / "metadata.json")
    soft_meta = load_metadata(args.soft_dir / "metadata.json")
    hard_rows = load_summary_csv(args.hard_dir / "gradient_summaries.csv")
    soft_rows = load_summary_csv(args.soft_dir / "gradient_summaries.csv")
    hard_steps, hard_coords, hard_vectors = load_sketches(args.hard_dir / "gradient_sketches.npz")
    soft_steps, soft_coords, soft_vectors = load_sketches(args.soft_dir / "gradient_sketches.npz")

    if hard_coords.shape != soft_coords.shape or not np.array_equal(hard_coords, soft_coords):
        raise ValueError("Hard and soft runs use different sketch coordinates; rerun with the same sketch seed/size.")

    per_layer_rows = per_layer_table(hard_rows, soft_rows)
    write_per_layer_csv(args.output_dir / "per_layer_sparsity.csv", per_layer_rows)

    hard_whole = np.asarray(
        [float(r["l1_over_sqrt_d_l2"]) for r in hard_rows.get("__whole_model__", [])],
        dtype=np.float64,
    )
    soft_whole = np.asarray(
        [float(r["l1_over_sqrt_d_l2"]) for r in soft_rows.get("__whole_model__", [])],
        dtype=np.float64,
    )
    sparsity_delta = bootstrap_mean_diff(hard_whole, soft_whole, args.bootstrap_samples, args.bootstrap_seed)

    hard_cosine = cosine_matrix(hard_vectors)
    soft_cosine = cosine_matrix(soft_vectors)
    hard_cos_values = upper_triangle_values(hard_cosine)
    soft_cos_values = upper_triangle_values(soft_cosine)
    coherence_delta = bootstrap_mean_diff(hard_cos_values, soft_cos_values, args.bootstrap_samples, args.bootstrap_seed)

    np.save(args.output_dir / "hard_whole_model_cosine.npy", hard_cosine)
    np.save(args.output_dir / "soft_whole_model_cosine.npy", soft_cosine)

    hard_rank = effective_rank(hard_vectors)
    soft_rank = effective_rank(soft_vectors)

    coherence_summary = {
        "hard": {
            "steps": hard_steps.tolist(),
            "num_pairs": int(hard_cos_values.size),
            "pairwise_cosine": summarize_metric(hard_cos_values),
            "effective_rank": hard_rank,
        },
        "soft": {
            "steps": soft_steps.tolist(),
            "num_pairs": int(soft_cos_values.size),
            "pairwise_cosine": summarize_metric(soft_cos_values),
            "effective_rank": soft_rank,
        },
        "delta": {
            "pairwise_cosine_mean": coherence_delta,
            "effective_rank": float(hard_rank["effective_rank"] - soft_rank["effective_rank"]),
            "stable_rank": float(hard_rank["stable_rank"] - soft_rank["stable_rank"]),
        },
    }
    with (args.output_dir / "coherence_summary.json").open("w", encoding="utf-8") as f:
        json.dump(coherence_summary, f, indent=2)
        f.write("\n")

    comparison_summary = {
        "hard": {"metadata": hard_meta},
        "soft": {"metadata": soft_meta},
        "whole_model_sparsity": {
            "hard": summarize_metric(hard_whole),
            "soft": summarize_metric(soft_whole),
            "delta": sparsity_delta,
        },
        "whole_model_coherence": coherence_summary,
    }
    with (args.output_dir / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)
        f.write("\n")

    print(f"Wrote gradient diagnostics comparison to {args.output_dir}")


if __name__ == "__main__":
    main()

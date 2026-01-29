#!/usr/bin/env python3
"""
Compare "weight update magnitude" across trained checkpoints by measuring how far each checkpoint's
parameters moved from a shared base model.

This is a post-hoc diagnostic: it does NOT require rerunning training.

We report several norms of (theta_finetuned - theta_base), aggregated over all floating-point tensors:
- l2: ||Δθ||_2
- rms: sqrt(mean(Δθ^2))  (scale-free per-parameter change)
- mean_abs: mean(|Δθ|)
- rel_l2: ||Δθ||_2 / ||θ_base||_2

Usage:
  python scripts/analysis/weight_update_magnitude.py \
    --base meta-llama/Llama-3.1-8B \
    --models checkpoints/llama3.1-8b-soft-panel checkpoints/llama3.1-8b-full-prism \
    --output artifacts/evaluations/diagnostics/weight_update_magnitude.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover
    snapshot_download = None  # type: ignore

try:
    from safetensors import safe_open
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "safetensors is required for this script. Install it via `pip install safetensors`."
    ) from exc


@dataclass(frozen=True)
class ModelTensors:
    root: Path
    weight_map: Dict[str, str]  # tensor_name -> shard_filename (relative to root)

    @property
    def keys(self) -> Set[str]:
        return set(self.weight_map.keys())


def _resolve_model_dir(model: str, hf_token: Optional[str]) -> Path:
    path = Path(model)
    if path.exists():
        return path
    if snapshot_download is None:
        raise RuntimeError(
            f"Model '{model}' is not a local path and huggingface_hub is not available."
        )
    local_dir = snapshot_download(repo_id=model, token=hf_token)
    return Path(local_dir)


def _load_safetensors_index(model_dir: Path) -> ModelTensors:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        weight_map = data.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"Malformed safetensors index: {index_path}")
        return ModelTensors(root=model_dir, weight_map={str(k): str(v) for k, v in weight_map.items()})

    # Fallback: enumerate *.safetensors and build a weight map by scanning keys.
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise RuntimeError(
            f"No safetensors found in {model_dir}. Expected model.safetensors.index.json or *.safetensors."
        )
    weight_map: Dict[str, str] = {}
    for file_path in files:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = file_path.name
    return ModelTensors(root=model_dir, weight_map=weight_map)


def _iter_unique_files(tensors: ModelTensors) -> Iterable[str]:
    return sorted(set(tensors.weight_map.values()))


def _read_global_step(model_dir: Path) -> Optional[int]:
    trainer_state = model_dir / "trainer_state.json"
    if not trainer_state.exists():
        return None
    try:
        with trainer_state.open("r", encoding="utf-8") as f:
            data = json.load(f)
        step = data.get("global_step")
        return int(step) if step is not None else None
    except Exception:
        return None


def _tensor_stats(
    base: ModelTensors,
    model: ModelTensors,
    *,
    show_progress: bool,
) -> Tuple[float, float, float, int]:
    """Return (sum_sq_diff, sum_abs_diff, sum_sq_base, n_elems) over all common floating tensors."""

    common_keys = base.keys & model.keys
    if not common_keys:
        raise RuntimeError("No overlapping tensors found between base and model.")

    # Open all shard files once using ExitStack.
    with ExitStack() as stack:
        base_handles = {
            fname: stack.enter_context(safe_open(base.root / fname, framework="pt", device="cpu"))
            for fname in _iter_unique_files(base)
        }
        model_handles = {
            fname: stack.enter_context(safe_open(model.root / fname, framework="pt", device="cpu"))
            for fname in _iter_unique_files(model)
        }

        sum_sq_diff = 0.0
        sum_abs_diff = 0.0
        sum_sq_base = 0.0
        n_elems = 0

        iterator = common_keys
        if show_progress:
            iterator = tqdm(sorted(common_keys), desc=f"Comparing {model.root.name}", unit="tensor")

        for key in iterator:
            base_file = base.weight_map[key]
            model_file = model.weight_map[key]
            b_handle = base_handles.get(base_file)
            m_handle = model_handles.get(model_file)
            if b_handle is None or m_handle is None:
                continue

            b = b_handle.get_tensor(key)
            t = m_handle.get_tensor(key)
            if not b.is_floating_point():
                continue
            if b.shape != t.shape:
                # Skip mismatched tensors (should not happen for full fine-tunes).
                continue

            # Compute in float32 for stable accumulation.
            b32 = b.float()
            d = (t.float() - b32)
            sum_sq_diff += float((d * d).sum().item())
            sum_abs_diff += float(d.abs().sum().item())
            sum_sq_base += float((b32 * b32).sum().item())
            n_elems += int(d.numel())

    if n_elems == 0:
        raise RuntimeError("No floating-point tensors were compared (n_elems == 0).")

    return sum_sq_diff, sum_abs_diff, sum_sq_base, n_elems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure parameter update magnitude vs a base model.")
    parser.add_argument("--base", required=True, help="Base model path or HF repo id.")
    parser.add_argument("--models", nargs="+", required=True, help="Fine-tuned model paths or HF repo ids.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Optional HF token for private models.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = _resolve_model_dir(args.base, args.hf_token)
    base_tensors = _load_safetensors_index(base_dir)

    results = []
    for model_name in args.models:
        model_dir = _resolve_model_dir(model_name, args.hf_token)
        model_tensors = _load_safetensors_index(model_dir)

        sum_sq_diff, sum_abs_diff, sum_sq_base, n_elems = _tensor_stats(
            base_tensors,
            model_tensors,
            show_progress=not args.no_progress,
        )

        l2 = math.sqrt(sum_sq_diff)
        base_l2 = math.sqrt(sum_sq_base)
        rms = math.sqrt(sum_sq_diff / n_elems)
        mean_abs = sum_abs_diff / n_elems
        rel_l2 = l2 / base_l2 if base_l2 > 0 else float("nan")

        results.append(
            {
                "model": model_name,
                "resolved_path": str(model_dir),
                "global_step": _read_global_step(model_dir),
                "n_elements": int(n_elems),
                "delta": {
                    "l2": l2,
                    "rms": rms,
                    "mean_abs": mean_abs,
                    "rel_l2": rel_l2,
                },
            }
        )

    out = {
        "base": args.base,
        "base_resolved_path": str(base_dir),
        "models": args.models,
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    print(f"Wrote weight-update diagnostics to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from dempo.utils import append_jsonl, read_jsonl
from dempo.wvs import COUNTRY_INFO, required_wvs_columns


OPTION_LABELS = ["A", "B", "C", "D"]


def safe_model_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def load_questions(path: Path) -> List[dict]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    return sorted(questions, key=lambda row: int(row["question_id"]))


def load_hf_model(model_id: str, hf_token: str | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_kwargs = {"token": hf_token}
    if "fix_mistral_regex" in inspect.signature(AutoTokenizer.from_pretrained).parameters:
        tok_kwargs["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def build_prompt(question: dict, system_prompt: str | None) -> str:
    option_lines = [f"({label}) {question['options'][label]}" for label in OPTION_LABELS]
    lines: List[str] = []
    system_prompt = (system_prompt or "").strip()
    if system_prompt:
        lines.append(f"System: {system_prompt}")
    lines.append(f"Human: {question['question_text']}")
    lines.append("")
    lines.append("Here are the options:")
    lines.append("")
    lines.extend(option_lines)
    lines.append("")
    lines.append("Assistant: If had to select one of the options, my answer would be (")
    return "\n".join(lines)


def option_token_sets(tokenizer) -> Dict[str, List[int]]:
    token_sets: Dict[str, List[int]] = {}
    for label in OPTION_LABELS:
        token_ids = set()
        for variant in [label, f" {label}", f"\n{label}"]:
            pieces = tokenizer.encode(variant, add_special_tokens=False)
            if len(pieces) == 1:
                token_ids.add(int(pieces[0]))
        if not token_ids:
            raise ValueError(
                f"Tokenizer for {getattr(tokenizer, 'name_or_path', 'model')} does not provide a "
                f"single-token encoding for option label {label!r}."
            )
        token_sets[label] = sorted(token_ids)
    return token_sets


def decode_top_tokens(tokenizer, token_ids: Iterable[int], probs: Iterable[float]) -> List[dict]:
    rows: List[dict] = []
    for token_id, prob in zip(token_ids, probs):
        rows.append(
            {
                "token_id": int(token_id),
                "token_str": tokenizer.decode([int(token_id)]),
                "probability": float(prob),
            }
        )
    return rows


def final_prompt_logits(logits: torch.Tensor, attention_mask: torch.Tensor, padding_side: str) -> torch.Tensor:
    if padding_side == "left":
        return logits[:, -1, :]
    last_positions = attention_mask.sum(dim=1) - 1
    return logits[torch.arange(logits.size(0), device=logits.device), last_positions]


def compute_question_probabilities(
    model,
    tokenizer,
    questions: Sequence[dict],
    batch_size: int,
    system_prompt: str | None,
    topk: int,
) -> List[dict]:
    prompts = [build_prompt(question, system_prompt) for question in questions]
    option_tokens = option_token_sets(tokenizer)
    rows: List[dict] = []
    for start in tqdm(range(0, len(questions), batch_size), desc="Scoring WVS question batches"):
        batch_questions = questions[start : start + batch_size]
        batch_prompts = prompts[start : start + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            logits = model(**inputs).logits
        last_logits = final_prompt_logits(
            logits,
            inputs["attention_mask"],
            getattr(tokenizer, "padding_side", "right"),
        ).to(dtype=torch.float32)
        log_probs = torch.log_softmax(last_logits, dim=-1)
        top_values, top_indices = torch.topk(log_probs, k=min(topk, log_probs.size(-1)), dim=-1)
        for idx, question in enumerate(batch_questions):
            raw_option_probs: Dict[str, float] = {}
            for label in OPTION_LABELS:
                token_ids = torch.tensor(option_tokens[label], device=log_probs.device)
                raw_prob = torch.exp(torch.logsumexp(log_probs[idx, token_ids], dim=0)).item()
                raw_option_probs[label] = float(raw_prob)
            option_total_probability_mass = float(sum(raw_option_probs.values()))
            if option_total_probability_mass > 0:
                normalized_option_probabilities = {
                    label: float(raw_option_probs[label] / option_total_probability_mass)
                    for label in OPTION_LABELS
                }
            else:
                normalized_option_probabilities = {label: 0.0 for label in OPTION_LABELS}
            rows.append(
                {
                    "question_id": int(question["question_id"]),
                    "question_code": question["question_code"],
                    "section": question["section"],
                    "question_text": question["question_text"],
                    "options": question["options"],
                    "scoring_position": "prompt_final_token",
                    "raw_option_probabilities": raw_option_probs,
                    "option_total_probability_mass": option_total_probability_mass,
                    "normalized_option_probabilities": normalized_option_probabilities,
                    "probabilities": normalized_option_probabilities,
                    "top_next_tokens": decode_top_tokens(
                        tokenizer,
                        top_indices[idx].detach().cpu().tolist(),
                        torch.exp(top_values[idx]).detach().cpu().tolist(),
                    ),
                }
            )
    return rows


def load_or_compute_question_probabilities(
    model_id: str,
    questions: Sequence[dict],
    output_path: Path,
    batch_size: int,
    hf_token: str | None,
    system_prompt: str | None,
    overwrite: bool,
    topk: int,
) -> List[dict]:
    if output_path.exists() and not overwrite:
        rows = read_jsonl(output_path)
        probe = rows[: min(len(rows), 8)]
        if rows and all(
            row.get("scoring_position") == "prompt_final_token"
            and "raw_option_probabilities" in row
            and "normalized_option_probabilities" in row
            and "option_total_probability_mass" in row
            and "top_next_tokens" in row
            for row in probe
        ):
            return rows
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_hf_model(model_id, hf_token)
    try:
        rows = compute_question_probabilities(
            model,
            tokenizer,
            questions=questions,
            batch_size=batch_size,
            system_prompt=system_prompt,
            topk=topk,
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    with output_path.open("w", encoding="utf-8") as handle:
        append_jsonl(handle, rows)
    return rows


def build_country_distributions(
    csv_path: Path,
    questions: Sequence[dict],
    min_respondents: int,
) -> List[dict]:
    df = pd.read_csv(csv_path, usecols=required_wvs_columns(questions))
    df["country_alpha"] = df["B_COUNTRY_ALPHA"].astype(str)
    df["country_weight"] = df["W_WEIGHT"].fillna(1.0)
    df["global_weight"] = df["W_WEIGHT"].fillna(1.0) * df["PWGHT"].fillna(1.0)

    rows: List[dict] = []
    for question in questions:
        column = question["question_code"]
        valid = df[column].isin([1, 2, 3, 4])
        question_df = df.loc[valid, ["country_alpha", "country_weight", "global_weight", column]].copy()
        if question_df.empty:
            continue
        for country_alpha, country_df in question_df.groupby("country_alpha"):
            respondent_count = int(len(country_df))
            if respondent_count < min_respondents:
                continue
            weighted = country_df.groupby(column)["country_weight"].sum()
            total = float(weighted.sum())
            if total <= 0:
                continue
            probs = {
                label: float(weighted.get(code, 0.0) / total)
                for label, code in question["value_codes"].items()
            }
            country_info = COUNTRY_INFO.get(country_alpha, {"country": country_alpha, "region": "Other"})
            rows.append(
                {
                    "country_alpha": country_alpha,
                    "country": country_info["country"],
                    "region": country_info["region"],
                    "question_id": int(question["question_id"]),
                    "question_code": column,
                    "respondents": respondent_count,
                    "probabilities": probs,
                    "weight_scheme": "country",
                }
            )
        global_weighted = question_df.groupby(column)["global_weight"].sum()
        global_total = float(global_weighted.sum())
        if global_total > 0:
            global_probs = {
                label: float(global_weighted.get(code, 0.0) / global_total)
                for label, code in question["value_codes"].items()
            }
            rows.append(
                {
                    "country_alpha": "GLOBAL",
                    "country": "Global aggregate",
                    "region": "Global",
                    "question_id": int(question["question_id"]),
                    "question_code": column,
                    "respondents": int(len(question_df)),
                    "probabilities": global_probs,
                    "weight_scheme": "global",
                }
            )
    return rows


def jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return float(np.sqrt(0.5 * (kl_pm + kl_qm)))


def summarize_model_against_countries(
    model_id: str,
    question_probs: Sequence[dict],
    country_distributions: Sequence[dict],
) -> List[dict]:
    model_map = {
        int(row["question_id"]): np.array(
            [
                (row.get("normalized_option_probabilities") or row["probabilities"])[label]
                for label in OPTION_LABELS
            ],
            dtype=float,
        )
        for row in question_probs
    }
    country_question_map: Dict[tuple[str, int], np.ndarray] = {}
    country_meta: Dict[str, dict] = {}
    for row in country_distributions:
        key = (str(row["country_alpha"]), int(row["question_id"]))
        country_question_map[key] = np.array([row["probabilities"][label] for label in OPTION_LABELS], dtype=float)
        country_meta[str(row["country_alpha"])] = {
            "country": row["country"],
            "region": row["region"],
            "weight_scheme": row["weight_scheme"],
        }

    summaries: List[dict] = []
    for country_alpha, meta in sorted(country_meta.items()):
        sims: List[float] = []
        l1s: List[float] = []
        for question_id, model_dist in model_map.items():
            ref_dist = country_question_map.get((country_alpha, question_id))
            if ref_dist is None:
                continue
            sims.append(1.0 - jensen_shannon_distance(model_dist, ref_dist))
            l1s.append(float(np.abs(model_dist - ref_dist).sum() / 2.0))
        if not sims:
            continue
        summaries.append(
            {
                "model": model_id,
                "country_alpha": country_alpha,
                "country": meta["country"],
                "region": meta["region"],
                "weight_scheme": meta["weight_scheme"],
                "mean_js_similarity": float(np.mean(sims)),
                "mean_total_variation": float(np.mean(l1s)),
                "questions_used": int(len(sims)),
            }
        )
    return sorted(summaries, key=lambda row: row["mean_js_similarity"], reverse=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate model option distributions on filtered WVS questions against country response distributions."
    )
    parser.add_argument(
        "--wvs-csv",
        type=Path,
        default=Path("wvs/WVS_Cross-National_Wave_7_csv_v6_0.csv"),
        help="Path to the WVS wave-7 CSV.",
    )
    parser.add_argument(
        "--questions-json",
        type=Path,
        default=Path("wvs/subjective_questions.json"),
        help="Path to the filtered subjective WVS question file.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Models to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for model probabilities and summaries.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for next-token scoring.",
    )
    parser.add_argument(
        "--min-respondents",
        type=int,
        default=100,
        help="Minimum valid respondents required for a country-question distribution.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face token for loading private models.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="",
        help="Optional system prompt injected ahead of each WVS question.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute model question probabilities even if cached outputs exist.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="How many top next tokens to save per question for diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_probs_dir = args.output_dir / "model_question_probs"
    model_probs_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(args.questions_json)
    country_distributions = build_country_distributions(
        csv_path=args.wvs_csv,
        questions=questions,
        min_respondents=args.min_respondents,
    )
    country_dist_path = args.output_dir / "country_distributions.jsonl"
    with country_dist_path.open("w", encoding="utf-8") as handle:
        append_jsonl(handle, country_distributions)

    all_summary_rows: List[dict] = []
    summary_json: Dict[str, dict] = {
        "question_count": len(questions),
        "min_respondents": args.min_respondents,
        "models": {},
    }
    for model_id in args.models:
        print(f"Evaluating WVS option probabilities for {model_id}")
        probs_path = model_probs_dir / f"{safe_model_name(model_id)}.jsonl"
        question_probs = load_or_compute_question_probabilities(
            model_id=model_id,
            questions=questions,
            output_path=probs_path,
            batch_size=args.batch_size,
            hf_token=args.hf_token,
            system_prompt=args.system_prompt,
            overwrite=args.overwrite,
            topk=args.topk,
        )
        summary_rows = summarize_model_against_countries(
            model_id=model_id,
            question_probs=question_probs,
            country_distributions=country_distributions,
        )
        all_summary_rows.extend(summary_rows)
        summary_json["models"][model_id] = {
            "top_countries": summary_rows[:10],
            "global": next((row for row in summary_rows if row["country_alpha"] == "GLOBAL"), None),
        }

    summary_df = pd.DataFrame(all_summary_rows)
    summary_csv_path = args.output_dir / "country_similarity.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    summary_json_path = args.output_dir / "summary.json"
    summary_json_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    print(f"Wrote WVS country distributions to {country_dist_path}")
    print(f"Wrote WVS similarity CSV to {summary_csv_path}")
    print(f"Wrote WVS summary JSON to {summary_json_path}")


if __name__ == "__main__":
    main()

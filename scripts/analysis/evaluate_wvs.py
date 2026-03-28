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

from dempo.utils import DEFAULT_CHAT_TEMPLATE, append_jsonl, build_chat_prompt, read_jsonl
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
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def build_messages(question: dict, system_prompt: str | None) -> List[dict]:
    option_lines = [f"{label}. {question['options'][label]}" for label in OPTION_LABELS]
    content = (
        "Please answer the following World Values Survey question by choosing exactly one option.\n\n"
        f"Question: {question['question_text']}\n\n"
        "Options:\n"
        + "\n".join(option_lines)
        + "\n\nRespond with a single capital letter: A, B, C, or D."
    )
    messages: List[dict] = []
    system_prompt = (system_prompt or "").strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    return messages


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


def compute_question_probabilities(
    model,
    tokenizer,
    questions: Sequence[dict],
    batch_size: int,
    system_prompt: str | None,
) -> List[dict]:
    prompts = [build_chat_prompt(tokenizer, build_messages(question, system_prompt)) for question in questions]
    option_tokens = option_token_sets(tokenizer)
    rows: List[dict] = []
    for start in tqdm(range(0, len(questions), batch_size), desc="Scoring WVS questions"):
        batch_questions = questions[start : start + batch_size]
        batch_prompts = prompts[start : start + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            logits = model(**inputs).logits
        last_positions = inputs["attention_mask"].sum(dim=1) - 1
        last_logits = logits[torch.arange(logits.size(0), device=logits.device), last_positions]
        for idx, question in enumerate(batch_questions):
            label_logits = []
            for label in OPTION_LABELS:
                token_ids = torch.tensor(option_tokens[label], device=last_logits.device)
                label_logits.append(torch.logsumexp(last_logits[idx, token_ids], dim=0))
            # Cast to fp32 before moving to CPU/NumPy; some PyTorch builds do not support
            # exporting CPU bfloat16 tensors directly to NumPy.
            label_logits_tensor = torch.stack(label_logits).to(dtype=torch.float32)
            label_probs = torch.softmax(label_logits_tensor, dim=0).detach().cpu().numpy()
            rows.append(
                {
                    "question_id": int(question["question_id"]),
                    "question_code": question["question_code"],
                    "section": question["section"],
                    "probabilities": {
                        label: float(label_probs[pos]) for pos, label in enumerate(OPTION_LABELS)
                    },
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
) -> List[dict]:
    if output_path.exists() and not overwrite:
        return read_jsonl(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_hf_model(model_id, hf_token)
    try:
        rows = compute_question_probabilities(
            model,
            tokenizer,
            questions=questions,
            batch_size=batch_size,
            system_prompt=system_prompt,
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
        int(row["question_id"]): np.array([row["probabilities"][label] for label in OPTION_LABELS], dtype=float)
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

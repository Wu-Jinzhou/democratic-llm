#!/usr/bin/env python3
"""
Constitutional evaluation pipeline:
- Load per-clause questions (generated via generate_questions.py)
- Query candidate models for answers
- Ask a judge model to rank answers (listwise) or pick a winner (pairwise)
- Optionally aggregate majority votes and write pairwise preference data

Candidate models are assumed to be Hugging Face causal LMs.
Judge model defaults to OpenAI gpt-5.2; can be overridden with --judge-model
to use a Hugging Face model instead.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import random
import re
import string
import time
import concurrent.futures
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
import inspect
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Ensure repo-local imports work even when executed from another working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from dempo.utils import (
    DEFAULT_CHAT_TEMPLATE,
    append_jsonl,
    build_chat_prompt,
    build_listwise_judge_prompts,
    build_pairwise_judge_prompts,
    seed_for_order,
    seed_for_question,
)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

def _decode_unicode_escapes(text: str) -> str:
    """Convert literal \\uXXXX sequences into unicode characters."""
    if "\\u" not in text and "\\U" not in text and "\\x" not in text:
        return text
    try:
        return text.encode("utf-8").decode("unicode_escape")
    except Exception:
        return text


def load_questions(
    directory: Path,
    max_questions: int | None = None,
    questions_per_clause: int | None = None,
    seed: int = 42,
) -> List[dict]:
    records: List[dict] = []
    for path in sorted(directory.glob("clause_*.json")):
        data = json.loads(path.read_text())
        questions = data["questions"]
        indices = list(range(len(questions)))
        if questions_per_clause is not None and len(indices) > questions_per_clause:
            clause_seed = seed + int(data["clause_id"])
            rng = random.Random(clause_seed)
            indices = rng.sample(indices, questions_per_clause)
        for idx in indices:
            q = _decode_unicode_escapes(str(questions[idx])).strip()
            records.append(
                {
                    "question_id": f"{data['clause_id']}:{idx}",
                    "clause_id": data["clause_id"],
                    "clause": data["clause"],
                    "question": q,
                }
            )
            if max_questions and len(records) >= max_questions:
                return records
    return records


def load_hf_model(model_id: str, hf_token: str | None):
    tok_kwargs = {"token": hf_token}
    if "fix_mistral_regex" in inspect.signature(AutoTokenizer.from_pretrained).parameters:
        tok_kwargs["fix_mistral_regex"] = True
    tok = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    if not getattr(tok, "chat_template", None):
        tok.chat_template = DEFAULT_CHAT_TEMPLATE
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 1.0
        model.generation_config.pad_token_id = tok.pad_token_id
    return model, tok


def _apply_stop_strings(text: str, stop_strings: List[str]) -> str:
    if not stop_strings:
        return text
    indices = [text.find(s) for s in stop_strings if s]
    indices = [idx for idx in indices if idx >= 0]
    if not indices:
        return text
    cut = min(indices)
    return text[:cut].rstrip()


def generate_answer(
    model,
    tokenizer,
    messages: List[dict],
    max_new_tokens: int,
    stop_strings: List[str],
) -> str:
    prompt = build_chat_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    text = text.strip()
    return _apply_stop_strings(text, stop_strings)


def generate_answers(
    model,
    tokenizer,
    messages_list: List[List[dict]],
    max_new_tokens: int,
    stop_strings: List[str],
) -> List[str]:
    prompts = [build_chat_prompt(tokenizer, messages) for messages in messages_list]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    generated = output[:, prompt_len:]
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return [_apply_stop_strings(text.strip(), stop_strings) for text in texts]


def judge_listwise_with_openai(
    client,
    judge_model: str,
    clause: str,
    question: str,
    labeled_answers: List[tuple[str, str]],
    max_output_tokens: int,
) -> Dict:
    system_prompt, user_prompt = build_listwise_judge_prompts(clause, question, labeled_answers)
    resp = client.responses.create(
        model=judge_model,
        input=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text={"format": {"type": "json_object"}},
        temperature=0,
        max_output_tokens=max_output_tokens,
    )
    content = resp.output_text or ""
    if not content.strip():
        raise ValueError("Judge model returned empty content.")
    return json.loads(content)


def judge_listwise_with_hf(
    model,
    tokenizer,
    clause: str,
    question: str,
    labeled_answers: List[tuple[str, str]],
    max_new_tokens: int,
) -> Dict:
    system_prompt, user_prompt = build_listwise_judge_prompts(clause, question, labeled_answers)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    output = generate_answer(model, tokenizer, messages, max_new_tokens=max_new_tokens, stop_strings=[])
    try:
        start = output.index("{")
        end = output.rindex("}") + 1
        return json.loads(output[start:end])
    except Exception:
        return {"ranking": None, "raw": output}


def judge_pairwise_with_openai(
    client,
    judge_model: str,
    clause: str,
    question: str,
    answer_a: str,
    answer_b: str,
    max_output_tokens: int,
) -> Dict:
    system_prompt, user_prompt = build_pairwise_judge_prompts(clause, question, answer_a, answer_b)
    resp = client.responses.create(
        model=judge_model,
        input=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text={"format": {"type": "json_object"}},
        temperature=0,
        max_output_tokens=max_output_tokens,
    )
    content = resp.output_text or ""
    if not content.strip():
        raise ValueError("Judge model returned empty content.")
    return json.loads(content)


def judge_pairwise_with_hf(
    model,
    tokenizer,
    clause: str,
    question: str,
    answer_a: str,
    answer_b: str,
    max_new_tokens: int,
) -> Dict:
    system_prompt, user_prompt = build_pairwise_judge_prompts(clause, question, answer_a, answer_b)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    output = generate_answer(model, tokenizer, messages, max_new_tokens=max_new_tokens, stop_strings=[])
    try:
        start = output.index("{")
        end = output.rindex("}") + 1
        return json.loads(output[start:end])
    except Exception:
        return {"winner": None, "raw": output}


def label_sequence(n: int) -> List[str]:
    labels = list(string.ascii_uppercase)
    if n <= len(labels):
        return labels[:n]
    out = labels[:]
    i = 0
    while len(out) < n:
        prefix = labels[i % len(labels)]
        for suffix in labels:
            out.append(f"{prefix}{suffix}")
            if len(out) >= n:
                break
        i += 1
    return out[:n]


def _normalize_label(raw: object, labels: List[str]) -> str:
    label_set = {label.upper() for label in labels}
    text = str(raw).strip().upper()
    text = re.sub(r"^(ANSWER|OPTION|CHOICE|MODEL)\s*[:\-]?\s*", "", text)
    text = text.strip().strip("()[]{}")
    text = text.rstrip(".):")
    if text in label_set:
        return text
    tokens = re.split(r"[\s,;]+", text)
    for token in [text] + tokens:
        candidate = token.strip().strip("()[]{}").rstrip(".):")
        if candidate in label_set:
            return candidate
    return text


def parse_ranking(payload: Dict, labels: List[str]) -> List[str]:
    ranking = payload.get("ranking")
    if not isinstance(ranking, list):
        raise ValueError(f"Invalid ranking payload: {payload}")
    normalized = [_normalize_label(item, labels) for item in ranking]
    label_set = {label.upper() for label in labels}
    if len(normalized) != len(labels):
        raise ValueError(f"Ranking missing labels or contains duplicates: {normalized}")
    if any(label not in label_set for label in normalized) or set(normalized) != label_set:
        raise ValueError(f"Ranking missing labels or contains duplicates: {normalized}")
    return normalized


def parse_winner(payload: Dict) -> str:
    winner = payload.get("winner")
    if isinstance(winner, str):
        candidate = winner.strip().upper()
        if candidate in {"A", "B"}:
            return candidate
    raw = payload.get("raw")
    if isinstance(raw, str):
        candidate = raw.strip().upper()
        if candidate.startswith("A"):
            return "A"
        if candidate.startswith("B"):
            return "B"
    raise ValueError(f"Invalid winner payload: {payload}")


def safe_model_id(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def load_existing_state(
    output_path: Path,
    preferences_path: Path,
    mode: str,
) -> tuple[set, set, dict, dict]:
    existing_questions: set[str] = set()
    existing_pref_pairs: set[tuple[str, str, str]] = set()
    existing_output_pairs: set[tuple[str, str, str]] = set()
    pref_counts: dict[str, int] = defaultdict(int)
    out_counts: dict[str, int] = defaultdict(int)

    if preferences_path.exists():
        with preferences_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (rec["question_id"], rec["model_i"], rec["model_j"])
                if key in existing_pref_pairs:
                    continue
                existing_pref_pairs.add(key)
                pref_counts[rec["question_id"]] += 1

    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if mode == "listwise":
                    existing_questions.add(rec["question_id"])
                else:
                    key = (rec["question_id"], rec["model_i"], rec["model_j"])
                    if key in existing_output_pairs:
                        continue
                    existing_output_pairs.add(key)
                    out_counts[rec["question_id"]] += 1

    return existing_questions, existing_pref_pairs, pref_counts, out_counts


def load_existing_listwise(output_path: Path) -> dict:
    existing: dict[str, dict] = {}
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                qid = rec.get("question_id")
                if not qid:
                    continue
                current = existing.get(qid)
                current_len = len(current.get("rankings", [])) if current else -1
                new_len = len(rec.get("rankings", []))
                if new_len >= current_len:
                    existing[qid] = rec
    return existing


def build_preferences_from_listwise(records: List[dict], model_pairs: List[tuple[str, str]]) -> List[dict]:
    preferences: List[dict] = []
    for rec in records:
        rankings = rec.get("rankings", [])
        if not rankings:
            continue
        question_id = rec["question_id"]
        clause_id = rec.get("clause_id")
        for model_i, model_j in model_pairs:
            wins_i = 0
            wins_j = 0
            for ranking in rankings:
                pos = {m: idx for idx, m in enumerate(ranking)}
                if pos[model_i] < pos[model_j]:
                    wins_i += 1
                else:
                    wins_j += 1
            majority_winner = None
            if wins_i > wins_j:
                majority_winner = model_i
            elif wins_j > wins_i:
                majority_winner = model_j
            preferences.append(
                {
                    "question_id": question_id,
                    "clause_id": clause_id,
                    "model_i": model_i,
                    "model_j": model_j,
                    "wins_i": wins_i,
                    "wins_j": wins_j,
                    "num_judges": len(rankings),
                    "majority_winner": majority_winner,
                }
            )
    return preferences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run constitutional evaluation with listwise or pairwise judging."
    )
    parser.add_argument("--questions-dir", type=Path, default=Path("artifacts/questions"))
    parser.add_argument("--mode", choices=["listwise", "pairwise"], default="listwise")
    parser.add_argument("--models", nargs="+", required=True, help="Candidate models; all pairs are compared.")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--judge-model", default="gpt-5.2", help="Judge model (OpenAI id or HF id).")
    parser.add_argument("--use-hf-judge", action="store_true", help="Force judge to run on HF instead of OpenAI.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--preferences-output", type=Path, default=Path("artifacts/evaluations/preferences.jsonl"))
    parser.add_argument("--responses-dir", type=Path, default=Path("artifacts/evaluations/responses"))
    parser.add_argument("--overwrite-responses", action="store_true")
    parser.add_argument(
        "--skip-judging",
        action="store_true",
        help="Only generate model responses and skip judging.",
    )
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--questions-per-clause", type=int, default=None)
    parser.add_argument("--num-judges", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--judge-max-output-tokens", type=int, default=400)
    parser.add_argument("--judge-retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument(
        "--judge-timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for each judge API call (OpenAI only).",
    )
    parser.add_argument(
        "--system-prompt",
        default="",
        help="Optional system prompt for candidate model generation. Default: empty (no system prompt).",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for local model generation.")
    parser.add_argument(
        "--judge-workers",
        type=int,
        default=1,
        help="Parallel workers for judging (OpenAI judge only).",
    )
    parser.add_argument(
        "--shuffle-order",
        action="store_true",
        default=True,
        help="Randomize model order when presenting to the judge (default).",
    )
    parser.add_argument(
        "--no-shuffle-order",
        action="store_false",
        dest="shuffle_order",
        help="Disable random order; present models in fixed order.",
    )
    parser.add_argument(
        "--stop-strings",
        nargs="*",
        default=["\nUser:", "\nHuman:"],
        help="Stop generation when any string is encountered.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    if args.output is None:
        args.output = Path(
            "artifacts/evaluations/listwise.jsonl"
            if args.mode == "listwise"
            else "artifacts/evaluations/pairwise.jsonl"
        )

    model_ids = args.models
    if len(model_ids) < 2:
        raise ValueError("Provide at least two models via --models.")

    questions = load_questions(
        args.questions_dir,
        max_questions=args.max_questions,
        questions_per_clause=args.questions_per_clause,
        seed=args.seed,
    )

    responses_dir = args.responses_dir
    responses_dir.mkdir(parents=True, exist_ok=True)
    responses_by_model: Dict[str, Dict[str, str]] = {}

    for model_id in model_ids:
        safe_id = safe_model_id(model_id)
        responses_path = responses_dir / f"{safe_id}.jsonl"
        model_responses: Dict[str, str] = {}
        if responses_path.exists() and not args.overwrite_responses:
            with responses_path.open("r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    model_responses[rec["question_id"]] = rec["response"]
            responses_by_model[model_id] = model_responses
            continue

        model, tokenizer = load_hf_model(model_id, args.hf_token)
        batch_size = max(1, args.batch_size)
        with responses_path.open("w", encoding="utf-8") as f:
            with tqdm(total=len(questions), desc=f"Generating responses ({model_id})") as pbar:
                for start in range(0, len(questions), batch_size):
                    batch = questions[start : start + batch_size]
                    messages_list: List[List[dict]] = []
                    for item in batch:
                        messages: List[dict] = []
                        if args.system_prompt:
                            messages.append({"role": "system", "content": args.system_prompt})
                        messages.append({"role": "user", "content": item["question"]})
                        messages_list.append(messages)
                    responses = generate_answers(
                        model,
                        tokenizer,
                        messages_list,
                        max_new_tokens=args.max_new_tokens,
                        stop_strings=args.stop_strings,
                    )
                    for item, response in zip(batch, responses):
                        model_responses[item["question_id"]] = response
                        f.write(
                            json.dumps(
                                {
                                    "question_id": item["question_id"],
                                    "clause_id": item["clause_id"],
                                    "question": item["question"],
                                    "response": response,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    pbar.update(len(batch))
        responses_by_model[model_id] = model_responses
        del model
        torch.cuda.empty_cache()

    if args.skip_judging:
        print("Skipping judging (responses generated).")
        return

    if args.use_hf_judge:
        judge_model, judge_tokenizer = load_hf_model(args.judge_model, args.hf_token)
        if args.mode == "listwise":
            judge_fn = lambda clause, q, labeled, max_tokens: judge_listwise_with_hf(
                judge_model, judge_tokenizer, clause, q, labeled, max_tokens
            )
        else:
            judge_fn = lambda clause, q, answer_a, answer_b, max_tokens: judge_pairwise_with_hf(
                judge_model, judge_tokenizer, clause, q, answer_a, answer_b, max_tokens
            )
    else:
        if OpenAI is None:
            raise ImportError("openai package not installed; install or use --use-hf-judge.")
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=args.judge_timeout)
        if args.mode == "listwise":
            judge_fn = lambda clause, q, labeled, max_tokens: judge_listwise_with_openai(
                client, args.judge_model, clause, q, labeled, max_tokens
            )
        else:
            judge_fn = lambda clause, q, answer_a, answer_b, max_tokens: judge_pairwise_with_openai(
                client, args.judge_model, clause, q, answer_a, answer_b, max_tokens
            )

    model_pairs = [(model_ids[i], model_ids[j]) for i in range(len(model_ids)) for j in range(i + 1, len(model_ids))]
    labels = label_sequence(len(model_ids))

    if args.use_hf_judge and args.judge_workers > 1:
        print("HF judge does not support parallel judging; using --judge-workers 1.")
        args.judge_workers = 1

    existing_listwise: dict[str, dict] = {}
    if args.mode == "listwise":
        existing_listwise = load_existing_listwise(args.output)
        existing_questions: set[str] = set()
        existing_pref_pairs: set[tuple[str, str, str]] = set()
        pref_counts: dict[str, int] = defaultdict(int)
        out_counts: dict[str, int] = defaultdict(int)
    else:
        existing_questions, existing_pref_pairs, pref_counts, out_counts = load_existing_state(
            args.output, args.preferences_output, args.mode
        )
    expected_pairs = len(model_pairs)
    pending: List[tuple[int, dict]] = []
    for idx, item in enumerate(questions, 1):
        qid = item["question_id"]
        if args.mode == "listwise":
            existing = existing_listwise.get(qid)
            if existing and len(existing.get("rankings", [])) >= args.num_judges:
                continue
        else:
            if pref_counts.get(qid, 0) >= expected_pairs and out_counts.get(qid, 0) >= expected_pairs:
                continue
        pending.append((idx, item))

    if not pending:
        if args.mode == "listwise":
            existing_listwise = load_existing_listwise(args.output)
            merged = list(existing_listwise.values())
            merged.sort(key=lambda r: (r.get("clause_id", 0), r.get("question_id", "")))
            with args.output.open("w", encoding="utf-8") as out_f:
                append_jsonl(out_f, merged)
            preferences = build_preferences_from_listwise(merged, model_pairs)
            with args.preferences_output.open("w", encoding="utf-8") as pref_f:
                append_jsonl(pref_f, preferences)
            print(
                f"No pending judgements; refreshed {len(merged)} listwise records and "
                f"{len(preferences)} preferences."
            )
        else:
            print("No pending judgements to run; outputs already complete.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.preferences_output.parent.mkdir(parents=True, exist_ok=True)

    def filter_new_preferences(records: List[dict]) -> List[dict]:
        if args.mode == "listwise":
            return records
        new_records: List[dict] = []
        for rec in records:
            key = (rec["question_id"], rec["model_i"], rec["model_j"])
            if key in existing_pref_pairs:
                continue
            existing_pref_pairs.add(key)
            pref_counts[rec["question_id"]] += 1
            new_records.append(rec)
        return new_records

    def filter_new_outputs(records: List[dict]) -> List[dict]:
        if args.mode == "listwise":
            return records
        new_records: List[dict] = []
        for rec in records:
            if args.mode == "listwise":
                qid = rec["question_id"]
                if qid in existing_questions:
                    continue
                existing_questions.add(qid)
                new_records.append(rec)
            else:
                key = (rec["question_id"], rec["model_i"], rec["model_j"])
                if key in existing_output_pairs:
                    continue
                existing_output_pairs.add(key)
                out_counts[rec["question_id"]] += 1
                new_records.append(rec)
        return new_records

    def process_question(item: dict, idx: int):
        clause = item["clause"]
        q = item["question"]
        question_id = item["question_id"]
        rng_local = random.Random(seed_for_question(question_id, args.seed))

        responses = []
        for model_id in model_ids:
            response = responses_by_model[model_id].get(question_id)
            if response is None:
                raise ValueError(f"Missing response for model {model_id} on {question_id}")
            responses.append((model_id, response))

        local_pref_records: List[dict] = []
        local_output_records: List[dict] = []

        if args.mode == "listwise":
            existing_record = existing_listwise.get(question_id)
            if existing_record is not None:
                existing_models = existing_record.get("models")
                if existing_models and list(existing_models) != list(model_ids):
                    raise ValueError(
                        f"Existing listwise record models do not match current --models for {question_id}."
                    )
            rankings: List[List[str]] = (
                list(existing_record.get("rankings", [])) if existing_record else []
            )
            raw_judgments: List[dict] = (
                list(existing_record.get("judge_raw", [])) if existing_record else []
            )
            judge_orders: List[List[str]] = (
                list(existing_record.get("judge_orders", [])) if existing_record else []
            )
            judge_label_maps: List[dict] = (
                list(existing_record.get("judge_label_to_model", [])) if existing_record else []
            )
            start_idx = len(rankings)
            for judge_idx in range(start_idx, args.num_judges):
                order = responses[:]
                if args.shuffle_order:
                    rng_judge = random.Random(seed_for_order(question_id, judge_idx, args.seed))
                    rng_judge.shuffle(order)
                labeled_answers = [(labels[i], order[i][1]) for i in range(len(order))]
                label_to_model = {labels[i]: order[i][0] for i in range(len(order))}
                attempt = 0
                while True:
                    try:
                        payload = judge_fn(clause, q, labeled_answers, args.judge_max_output_tokens)
                        ranking_labels = parse_ranking(payload, labels)
                        ranking_models = [label_to_model[label] for label in ranking_labels]
                        rankings.append(ranking_models)
                        raw_judgments.append(payload)
                        judge_orders.append([model_id for model_id, _ in order])
                        judge_label_maps.append(label_to_model)
                        break
                    except Exception as exc:
                        attempt += 1
                        if attempt > args.judge_retries:
                            raise RuntimeError(f"Judge failed after retries: {exc}") from exc
                        sleep_for = args.retry_backoff ** attempt
                        print(
                            f"[judge retry] mode=listwise question_id={question_id} "
                            f"judge_idx={judge_idx} attempt={attempt}/{args.judge_retries} "
                            f"sleep={sleep_for:.1f}s error={exc}",
                            flush=True,
                        )
                        time.sleep(sleep_for)

            for model_i, model_j in model_pairs:
                wins_i = 0
                wins_j = 0
                for ranking in rankings:
                    pos = {m: idx for idx, m in enumerate(ranking)}
                    if pos[model_i] < pos[model_j]:
                        wins_i += 1
                    else:
                        wins_j += 1
                majority_winner = None
                if wins_i > wins_j:
                    majority_winner = model_i
                elif wins_j > wins_i:
                    majority_winner = model_j
                local_pref_records.append(
                    {
                        "question_id": question_id,
                        "clause_id": item["clause_id"],
                        "model_i": model_i,
                        "model_j": model_j,
                        "wins_i": wins_i,
                        "wins_j": wins_j,
                        "num_judges": args.num_judges,
                        "majority_winner": majority_winner,
                    }
                )

            local_output_records.append(
                {
                    "idx": idx,
                    "question_id": question_id,
                    "clause_id": item["clause_id"],
                    "clause": clause,
                    "question": q,
                    "models": model_ids,
                    "responses": {model_id: resp for model_id, resp in responses},
                    "rankings": rankings,
                    "judge_raw": raw_judgments,
                    "judge_orders": judge_orders,
                    "judge_label_to_model": judge_label_maps,
                }
            )
        else:
            for model_i, model_j in model_pairs:
                response_i = responses_by_model[model_i].get(question_id)
                response_j = responses_by_model[model_j].get(question_id)
                if response_i is None:
                    raise ValueError(f"Missing response for model {model_i} on {question_id}")
                if response_j is None:
                    raise ValueError(f"Missing response for model {model_j} on {question_id}")

                wins_i = 0
                wins_j = 0
                raw_judgments: List[dict] = []
                for judge_idx in range(args.num_judges):
                    order = [(model_i, response_i), (model_j, response_j)]
                    if args.shuffle_order:
                        rng_judge = random.Random(
                            seed_for_order(question_id, judge_idx, args.seed, f"{model_i}:{model_j}")
                        )
                        rng_judge.shuffle(order)
                    label_to_model = {"A": order[0][0], "B": order[1][0]}
                    answer_a = order[0][1]
                    answer_b = order[1][1]
                    order_models = [order[0][0], order[1][0]]
                    attempt = 0
                    while True:
                        try:
                            payload = judge_fn(
                                clause, q, answer_a, answer_b, args.judge_max_output_tokens
                            )
                            winner_label = parse_winner(payload)
                            winner_model = label_to_model[winner_label]
                            if winner_model == model_i:
                                wins_i += 1
                            else:
                                wins_j += 1
                            raw_judgments.append(
                                {
                                    "winner_label": winner_label,
                                    "label_to_model": label_to_model,
                                    "presentation_order": order_models,
                                    "payload": payload,
                                }
                            )
                            break
                        except Exception as exc:
                            attempt += 1
                            if attempt > args.judge_retries:
                                raise RuntimeError(f"Judge failed after retries: {exc}") from exc
                            sleep_for = args.retry_backoff ** attempt
                            print(
                                f"[judge retry] mode=pairwise question_id={question_id} "
                                f"pair={model_i}__vs__{model_j} attempt={attempt}/{args.judge_retries} "
                                f"sleep={sleep_for:.1f}s error={exc}",
                                flush=True,
                            )
                            time.sleep(sleep_for)

                majority_winner = None
                if wins_i > wins_j:
                    majority_winner = model_i
                elif wins_j > wins_i:
                    majority_winner = model_j

                local_pref_records.append(
                    {
                        "question_id": question_id,
                        "clause_id": item["clause_id"],
                        "model_i": model_i,
                        "model_j": model_j,
                        "wins_i": wins_i,
                        "wins_j": wins_j,
                        "num_judges": args.num_judges,
                        "majority_winner": majority_winner,
                    }
                )

                local_output_records.append(
                    {
                        "idx": idx,
                        "question_id": question_id,
                        "clause_id": item["clause_id"],
                        "clause": clause,
                        "question": q,
                        "model_i": model_i,
                        "model_j": model_j,
                        "responses": {
                            model_i: response_i,
                            model_j: response_j,
                        },
                        "wins_i": wins_i,
                        "wins_j": wins_j,
                        "num_judges": args.num_judges,
                        "majority_winner": majority_winner,
                        "judge_raw": raw_judgments,
                    }
                )
        return local_pref_records, local_output_records

    with args.output.open("a", encoding="utf-8") as out_f, args.preferences_output.open(
        "a", encoding="utf-8"
    ) as pref_f:
        failures: List[dict] = []
        if args.judge_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.judge_workers) as ex:
                futures = {
                    ex.submit(process_question, item, idx): (idx, item["question_id"])
                    for idx, item in pending
                }
                for fut in tqdm(
                    concurrent.futures.as_completed(futures), total=len(futures), desc="Judging"
                ):
                    idx, qid = futures[fut]
                    try:
                        pref, out = fut.result()
                    except Exception as exc:
                        failures.append({"idx": idx, "question_id": qid, "error": str(exc)})
                        print(f"Judging failed for {qid}: {exc}", file=sys.stderr)
                        continue
                    if args.mode == "listwise":
                        if out:
                            rec = out[0]
                            existing_listwise[rec["question_id"]] = rec
                            append_jsonl(out_f, out)
                    else:
                        new_pref = filter_new_preferences(pref)
                        new_out = filter_new_outputs(out)
                        if new_pref:
                            append_jsonl(pref_f, new_pref)
                        if new_out:
                            append_jsonl(out_f, new_out)
        else:
            for idx, item in tqdm(pending, desc="Judging"):
                try:
                    pref, out = process_question(item, idx)
                except Exception as exc:
                    failures.append({"idx": idx, "question_id": item["question_id"], "error": str(exc)})
                    print(f"Judging failed for {item['question_id']}: {exc}", file=sys.stderr)
                    continue
                if args.mode == "listwise":
                    if out:
                        rec = out[0]
                        existing_listwise[rec["question_id"]] = rec
                        append_jsonl(out_f, out)
                else:
                    new_pref = filter_new_preferences(pref)
                    new_out = filter_new_outputs(out)
                    if new_pref:
                        append_jsonl(pref_f, new_pref)
                    if new_out:
                        append_jsonl(out_f, new_out)

    if args.mode == "listwise":
        merged = list(existing_listwise.values())
        merged.sort(key=lambda r: (r.get("clause_id", 0), r.get("question_id", "")))
        with args.output.open("w", encoding="utf-8") as out_f:
            append_jsonl(out_f, merged)
        preferences = build_preferences_from_listwise(merged, model_pairs)
        with args.preferences_output.open("w", encoding="utf-8") as pref_f:
            append_jsonl(pref_f, preferences)
        print(
            f"Wrote {len(merged)} listwise records and {len(preferences)} preferences to "
            f"{args.output} and {args.preferences_output}."
        )
        if failures:
            failures_path = args.output.with_suffix(".failures.jsonl")
            with failures_path.open("w", encoding="utf-8") as f:
                append_jsonl(f, failures)
            raise SystemExit(
                f"Completed with {len(failures)} judging failures. See {failures_path}."
            )


if __name__ == "__main__":
    main()

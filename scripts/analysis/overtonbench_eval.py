#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

import pandas as pd
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
for path in (REPO_ROOT,):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from dempo.utils import DEFAULT_CHAT_TEMPLATE, append_jsonl, build_chat_prompt
from third_party.overtonbench import helper_functions, llm_api, prompts
from third_party.overtonbench.scoring import compute_unadjusted_overton, compute_weighted_overton


PROMPT_TYPE_MAP = {
    "fr": "freeresponse",
    "demog": "demog",
    "demog+fr": "demog_freeresponse",
    "fs": "fewshot",
    "fs+fr": "freeresponse_fewshot",
}


def safe_model_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def chunked(seq: Sequence, size: int) -> Iterator[Sequence]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_questions(path: Path, source: str) -> pd.DataFrame:
    questions = pd.read_csv(path)
    questions["question_id"] = pd.to_numeric(questions["question_id"], errors="raise").astype(int)
    source = source.strip().lower()
    if source != "full":
        questions = questions[questions["source"].str.lower() == source].copy()
    return questions.sort_values(["source", "question_id"]).reset_index(drop=True)


def load_hf_model(model_id: str, hf_token: str | None):
    import inspect
    import torch
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
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 1.0
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def generate_answers(
    model,
    tokenizer,
    messages_list: List[List[dict]],
    max_new_tokens: int,
) -> List[str]:
    import torch

    prompts_list = [build_chat_prompt(tokenizer, messages) for messages in messages_list]
    inputs = tokenizer(prompts_list, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    generated = output[:, prompt_len:]
    return [text.strip() for text in tokenizer.batch_decode(generated, skip_special_tokens=True)]


def load_response_map(path: Path) -> Dict[int, str]:
    rows = read_jsonl(path)
    out: Dict[int, str] = {}
    for row in rows:
        question_id = int(row["question_id"])
        out[question_id] = str(row["response"])
    return out


def build_generation_messages(question: str, system_prompt: str | None) -> List[dict]:
    messages: List[dict] = []
    system_prompt = (system_prompt or "").strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    return messages


def run_generate(args: argparse.Namespace) -> None:
    questions = load_questions(args.questions_csv, args.source)
    args.responses_dir.mkdir(parents=True, exist_ok=True)
    for model_id in args.models:
        response_path = args.responses_dir / f"{safe_model_name(model_id)}.jsonl"
        completed = {int(row["question_id"]) for row in read_jsonl(response_path)}
        pending = questions[~questions["question_id"].isin(completed)].copy()
        if pending.empty and not args.overwrite:
            print(f"Skipping generation for {model_id}; all {len(questions)} questions already cached.")
            continue
        if args.overwrite and response_path.exists():
            response_path.unlink()
            pending = questions.copy()
        print(f"Loading model: {model_id}")
        model, tokenizer = load_hf_model(model_id, args.hf_token)
        try:
            with response_path.open("a", encoding="utf-8") as handle:
                for batch_df in tqdm(
                    list(chunked(list(pending.to_dict(orient="records")), args.batch_size)),
                    desc=f"Generating {model_id}",
                ):
                    messages_list = [
                        build_generation_messages(row["question_text"], args.system_prompt)
                        for row in batch_df
                    ]
                    texts = generate_answers(model, tokenizer, messages_list, args.max_new_tokens)
                    records = []
                    for row, text in zip(batch_df, texts):
                        records.append(
                            {
                                "question_id": int(row["question_id"]),
                                "source": row["source"],
                                "question": row["question_text"],
                                "model": model_id,
                                "response": text,
                            }
                        )
                    append_jsonl(handle, records)
        finally:
            del model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def load_benchmark_rows(benchmark_csv: Path, questions_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(benchmark_csv)
    df["question_id"] = pd.to_numeric(df["question_id"], errors="raise").astype(int)
    question_meta = load_questions(questions_csv, "full")[["question_id", "source"]]
    merged = df.merge(question_meta, on="question_id", how="left")
    if merged["source"].isna().any():
        missing = sorted(merged.loc[merged["source"].isna(), "question_id"].unique().tolist())
        raise ValueError(f"Missing source metadata for question ids: {missing[:10]}")
    return merged


def load_completed_keys(path: Path) -> set[tuple[str, int]]:
    completed: set[tuple[str, int]] = set()
    for row in read_jsonl(path):
        completed.add((str(row["user"]), int(row["question_id"])))
    return completed


def score_row(
    row: dict,
    prompt_type: str,
    judge_model: str,
) -> dict:
    formatted_prompt = helper_functions.format_prompt(prompt_type, row)
    rating = llm_api.generate_rating(
        preformatted_prompt=formatted_prompt,
        sys_prompt=prompts.system_prompt,
        model_name=judge_model,
        temperature=0,
    )
    if rating is None:
        raise RuntimeError(
            f"Judge returned no rating for user={row['user']} question_id={row['question_id']} model={row['model']}"
        )
    return {
        "user": str(row["user"]),
        "question_id": int(row["question_id"]),
        "source": row["source"],
        "model": row["model"],
        "cluster_kmeans": row["cluster_kmeans"],
        "representation_rating_pred": int(rating),
        "judge_model": judge_model,
        "prompt_type": prompt_type,
    }


def run_judge(args: argparse.Namespace) -> None:
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        raise EnvironmentError("Set GEMINI_API_KEY or GOOGLE_API_KEY before running OvertonBench judging.")
    prompt_type = PROMPT_TYPE_MAP[args.prompt]
    helper_functions.set_data_path(str(args.benchmark_csv))
    benchmark_rows = load_benchmark_rows(args.benchmark_csv, args.questions_csv)
    args.predictions_dir.mkdir(parents=True, exist_ok=True)
    try:
        for model_id in args.models:
            response_path = args.responses_dir / f"{safe_model_name(model_id)}.jsonl"
            if not response_path.exists():
                raise FileNotFoundError(f"Missing responses for {model_id}: {response_path}")
            response_map = load_response_map(response_path)
            output_path = args.predictions_dir / f"{safe_model_name(model_id)}.jsonl"
            completed = load_completed_keys(output_path)
            pending_rows = []
            for row in benchmark_rows.to_dict(orient="records"):
                qid = int(row["question_id"])
                response = response_map.get(qid)
                if response is None:
                    continue
                key = (str(row["user"]), qid)
                if key in completed:
                    continue
                row = dict(row)
                row["model"] = model_id
                row["llm_response"] = response
                pending_rows.append(row)
            print(
                f"Scoring {model_id}: {len(completed)} cached, {len(pending_rows)} pending rows "
                f"with {args.judge_model} and prompt {args.prompt}."
            )
            if not pending_rows:
                continue
            with output_path.open("a", encoding="utf-8") as handle:
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                    futures = {
                        pool.submit(score_row, row, prompt_type, args.judge_model): row
                        for row in pending_rows
                    }
                    for future in tqdm(
                        concurrent.futures.as_completed(futures),
                        total=len(futures),
                        desc=f"Judging {safe_model_name(model_id)}",
                    ):
                        row = futures[future]
                        try:
                            scored = future.result()
                        except Exception as exc:
                            raise RuntimeError(
                                f"Failed scoring user={row['user']} question_id={row['question_id']} "
                                f"model={model_id}: {exc}"
                            ) from exc
                        append_jsonl(handle, [scored])
    finally:
        llm_api.close_gemini_client()


def load_prediction_frame(predictions_dir: Path, models: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for model_id in models:
        path = predictions_dir / f"{safe_model_name(model_id)}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction file for {model_id}: {path}")
        rows = read_jsonl(path)
        frame = pd.DataFrame(rows)
        if frame.empty:
            continue
        frame["model"] = model_id
        frame["question_id"] = pd.to_numeric(frame["question_id"], errors="raise").astype(int)
        frame["representation_rating"] = pd.to_numeric(
            frame["representation_rating_pred"], errors="coerce"
        )
        frames.append(frame)
    if not frames:
        raise ValueError("No predictions found to summarize.")
    return pd.concat(frames, ignore_index=True)


def summarize_split(df: pd.DataFrame, tau: float, cluster_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "OvertonScore",
                "total_clusters",
                "avg_clusters_per_q",
                "OvertonScoreW",
                "n_rows",
                "n_users",
                "n_questions",
            ]
        )
    raw_scores, _ = compute_unadjusted_overton(df, cluster_col=cluster_col, tau=tau)
    weighted_scores, _ = compute_weighted_overton(df, cluster_col=cluster_col, tau=tau)
    merged = raw_scores.merge(weighted_scores, on="model", how="outer")
    merged = merged.rename(columns={"OvertonScore_w": "OvertonScoreW"})
    counts = (
        df.groupby("model")
        .agg(
            n_rows=("user", "size"),
            n_users=("user", "nunique"),
            n_questions=("question_id", "nunique"),
        )
        .reset_index()
    )
    return merged.merge(counts, on="model", how="left")


def run_summarize(args: argparse.Namespace) -> None:
    predictions = load_prediction_frame(args.predictions_dir, args.models)
    split_frames = {
        "overall": predictions.copy(),
        "prism": predictions[predictions["source"].str.lower() == "prism"].copy(),
        "modelslant": predictions[predictions["source"].str.lower() == "modelslant"].copy(),
    }
    flat_rows: List[dict] = []
    nested: List[dict] = []
    for model_id in args.models:
        model_result = {
            "model": model_id,
            "judge_model": args.judge_model,
            "prompt": args.prompt,
            "tau": args.tau,
        }
        for split_name, split_df in split_frames.items():
            table = summarize_split(split_df, tau=args.tau, cluster_col=args.cluster_col)
            row = table[table["model"] == model_id]
            if row.empty:
                metrics = {}
            else:
                record = row.iloc[0].to_dict()
                metrics = {
                    "OvertonScore": float(record["OvertonScore"]),
                    "OvertonScoreW": float(record["OvertonScoreW"]),
                    "total_clusters": int(record["total_clusters"]),
                    "avg_clusters_per_q": float(record["avg_clusters_per_q"]),
                    "n_rows": int(record["n_rows"]),
                    "n_users": int(record["n_users"]),
                    "n_questions": int(record["n_questions"]),
                }
                flat_rows.append(
                    {
                        "model": model_id,
                        "split": split_name,
                        **metrics,
                    }
                )
            model_result[split_name] = metrics
        nested.append(model_result)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(
            {
                "judge_model": args.judge_model,
                "prompt": args.prompt,
                "tau": args.tau,
                "models": nested,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(flat_rows).to_csv(args.summary_csv, index=False)
    print(f"Wrote summary JSON to {args.summary_json}")
    print(f"Wrote summary CSV to {args.summary_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate local models on OvertonBench with Gemini FS+FR.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = {
        "questions_csv": Path("/overtonbench/meta/questions.csv"),
        "benchmark_csv": Path("/overtonbench/data/prolific_with_clusters_kmeans_merged_public.csv"),
    }

    gen = subparsers.add_parser("generate", help="Generate model responses for OvertonBench questions.")
    gen.add_argument("--questions-csv", type=Path, default=common["questions_csv"])
    gen.add_argument("--source", choices=["full", "prism", "modelslant"], default="full")
    gen.add_argument("--models", nargs="+", required=True)
    gen.add_argument("--responses-dir", type=Path, required=True)
    gen.add_argument("--batch-size", type=int, default=8)
    gen.add_argument("--max-new-tokens", type=int, default=256)
    gen.add_argument("--system-prompt", default="")
    gen.add_argument("--hf-token", default=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"))
    gen.add_argument("--overwrite", action="store_true")
    gen.set_defaults(func=run_generate)

    judge = subparsers.add_parser("judge", help="Score model responses with Gemini using OvertonBench FS+FR.")
    judge.add_argument("--questions-csv", type=Path, default=common["questions_csv"])
    judge.add_argument("--benchmark-csv", type=Path, default=common["benchmark_csv"])
    judge.add_argument("--responses-dir", type=Path, required=True)
    judge.add_argument("--predictions-dir", type=Path, required=True)
    judge.add_argument("--models", nargs="+", required=True)
    judge.add_argument("--judge-model", default="gemini-2.5-pro")
    judge.add_argument("--prompt", choices=sorted(PROMPT_TYPE_MAP.keys()), default="fs+fr")
    judge.add_argument("--max-workers", type=int, default=8)
    judge.set_defaults(func=run_judge)

    summ = subparsers.add_parser("summarize", help="Aggregate OvertonBench scores from prediction files.")
    summ.add_argument("--predictions-dir", type=Path, required=True)
    summ.add_argument("--models", nargs="+", required=True)
    summ.add_argument("--summary-json", type=Path, required=True)
    summ.add_argument("--summary-csv", type=Path, required=True)
    summ.add_argument("--judge-model", default="gemini-2.5-pro")
    summ.add_argument("--prompt", choices=sorted(PROMPT_TYPE_MAP.keys()), default="fs+fr")
    summ.add_argument("--tau", type=float, default=4.0)
    summ.add_argument("--cluster-col", default="cluster_kmeans")
    summ.set_defaults(func=run_summarize)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    start = time.time()
    args.func(args)
    elapsed = (time.time() - start) / 60.0
    print(f"Completed {args.command} in {elapsed:.2f} minutes.")


if __name__ == "__main__":
    main()

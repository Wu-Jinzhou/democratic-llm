#!/usr/bin/env python3
"""
DPO fine-tuning for meta-llama/Llama-3.1-8B with hard/soft panel weights.

Requires:
- Hugging Face access to meta-llama/Llama-3.1-8B (accept license + set HF_TOKEN or --hf-token)
- datasets, transformers, trl, torch installed
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig


@dataclass
class TrainConfig:
    model_id: str
    output_dir: Path
    dataset_path: Path
    hf_token: Optional[str]
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
    num_train_epochs: float = 1.0
    beta: float = 0.1
    weight_decay: float = 0.0
    eval_ratio: float = 0.02
    seed: int = 0


class WeightedDPOTrainer(DPOTrainer):
    """DPO trainer that supports per-example weights (column 'weight')."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        weights = inputs.pop("weight", None)
        result = self.get_batch_loss(model, inputs, train_eval="train")
        if isinstance(result, (list, tuple)):
            loss = result[0]
            metrics = result[1] if len(result) > 1 else None
        else:
            loss, metrics = result, None
        if weights is not None:
            weight_tensor = weights.to(loss.device)
            # Reduce with weights; loss from TRL is already per-sample
            loss = (loss * weight_tensor).mean()
        if return_outputs:
            return loss, metrics
        return loss


def load_tokenizer(model_id: str, token: Optional[str]):
    tok = AutoTokenizer.from_pretrained(model_id, token=token, use_auth_token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_model(model_id: str, token: Optional[str]):
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        use_auth_token=token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )


def build_datasets(path: Path, eval_ratio: float):
    dataset = datasets.load_dataset("json", data_files=str(path))["train"]
    dataset = dataset.shuffle(seed=0)
    if eval_ratio and eval_ratio > 0:
        split = dataset.train_test_split(test_size=eval_ratio, seed=0)
        return split["train"], split["test"]
    return dataset, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Llama-3.1-8B with DPO on PRISM-prepared data.")
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL with prompt/chosen/rejected (+ optional weight).")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/llama3.1-8b-dpo"))
    parser.add_argument("--model-id", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        model_id=args.model_id,
        output_dir=args.output_dir,
        dataset_path=args.dataset,
        hf_token=args.hf_token,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        beta=args.beta,
        weight_decay=args.weight_decay,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )

    tokenizer = load_tokenizer(cfg.model_id, cfg.hf_token)
    model = load_model(cfg.model_id, cfg.hf_token)
    ref_model = load_model(cfg.model_id, cfg.hf_token)

    train_ds, eval_ds = build_datasets(cfg.dataset_path, cfg.eval_ratio)
    training_args = DPOConfig(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=10,
        save_steps=500,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=500,
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=True,
        max_length=2048,
        max_prompt_length=1024,
        weight_decay=cfg.weight_decay,
        seed=cfg.seed,
    )

    trainer_cls = WeightedDPOTrainer if "weight" in train_ds.column_names else DPOTrainer
    trainer = trainer_cls(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=cfg.beta,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))


if __name__ == "__main__":
    main()

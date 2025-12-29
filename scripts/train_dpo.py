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
from trl import DPOTrainer, DPOConfig, DataCollatorForPreference
from tqdm import tqdm


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
    seed: int = 42
    report_to: str = "wandb"
    logging_dir: Path = Path("logs")
    run_name: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None


class WeightedDataCollatorForPreference(DataCollatorForPreference):
    """Data collator that preserves per-example weights."""

    def torch_call(self, examples):
        output = super().torch_call(examples)
        if "weight" in examples[0]:
            output["weight"] = torch.tensor(
                [example["weight"] for example in examples],
                dtype=torch.float32,
            )
        return output


class WeightedDPOTrainer(DPOTrainer):
    """DPO trainer that supports per-example weights (column 'weight')."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._batch_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        weights = inputs.pop("weight", None)
        self._batch_weights = weights
        try:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
        finally:
            self._batch_weights = None

    def dpo_loss(
        self,
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        loss_type="sigmoid",
        model_output=None,
    ):
        losses, chosen_rewards, rejected_rewards = super().dpo_loss(
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            loss_type=loss_type,
            model_output=model_output,
        )
        if self._batch_weights is not None:
            weight_tensor = self._batch_weights.to(losses.device).float()
            scale = weight_tensor.numel() / weight_tensor.sum().clamp(min=1e-8)
            losses = losses * weight_tensor * scale
        return losses, chosen_rewards, rejected_rewards


def load_tokenizer(model_id: str, token: Optional[str]):
    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_model(model_id: str, token: Optional[str]):
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )


def build_datasets(path: Path, eval_ratio: float, seed: int):
    dataset = datasets.load_dataset("json", data_files=str(path))["train"]
    dataset = dataset.shuffle(seed=seed)
    if eval_ratio and eval_ratio > 0:
        split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", default="wandb", help="Logging backend (wandb, tensorboard, or none).")
    parser.add_argument("--logging-dir", type=Path, default=Path("logs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
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
        report_to=args.report_to,
        logging_dir=args.logging_dir,
        run_name=args.run_name,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
    )

    if cfg.report_to == "wandb":
        try:
            import wandb  # noqa: F401
        except ImportError as exc:
            raise ImportError("W&B requested but not installed. Run: pip install wandb") from exc
        if cfg.wandb_project:
            os.environ["WANDB_PROJECT"] = cfg.wandb_project
        if cfg.wandb_entity:
            os.environ["WANDB_ENTITY"] = cfg.wandb_entity
        if cfg.wandb_group:
            os.environ["WANDB_RUN_GROUP"] = cfg.wandb_group

    tokenizer = load_tokenizer(cfg.model_id, cfg.hf_token)
    model = load_model(cfg.model_id, cfg.hf_token)
    ref_model = load_model(cfg.model_id, cfg.hf_token)

    train_ds, eval_ds = build_datasets(cfg.dataset_path, cfg.eval_ratio, cfg.seed)
    print(f"Loaded dataset: {len(train_ds)} train rows" + (f", {len(eval_ds)} eval rows" if eval_ds else ""))
    report_to = [] if cfg.report_to == "none" else [cfg.report_to]
    dpo_kwargs = dict(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps" if eval_ds else "no",
        eval_steps=500,
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=True,
        max_length=2048,
        max_prompt_length=1024,
        weight_decay=cfg.weight_decay,
        seed=cfg.seed,
        report_to=report_to,
        logging_dir=str(cfg.logging_dir),
        run_name=cfg.run_name,
        beta=cfg.beta,
    )
    training_args = DPOConfig(**dpo_kwargs)

    trainer_cls = WeightedDPOTrainer if "weight" in train_ds.column_names else DPOTrainer
    data_collator = WeightedDataCollatorForPreference(pad_token_id=tokenizer.pad_token_id)
    trainer = trainer_cls(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))


if __name__ == "__main__":
    main()

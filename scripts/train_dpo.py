#!/usr/bin/env python3
"""
DPO fine-tuning for meta-llama/Llama-3.1-8B with hard/soft panel weights.

Requires:
- Hugging Face access to meta-llama/Llama-3.1-8B (accept license + set HF_TOKEN or --hf-token)
- datasets, transformers, trl, torch installed
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from trl.trainer.dpo_trainer import DataCollatorForPreference

# Ensure repo-local imports work even when executed from another working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from dempo.utils import DEFAULT_CHAT_TEMPLATE


@dataclass
class TrainConfig:
    model_id: str
    output_dir: Path
    dataset_path: Path
    hf_token: Optional[str]
    device_map: Optional[str] = "auto"
    attn_implementation: Optional[str] = "flash_attention_2"
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
    num_train_epochs: float = 2.0
    num_train_steps: int = -1
    beta: float = 0.1
    weight_decay: float = 0.0
    eval_ratio: float = 0.02
    eval_strategy: str = "steps"
    eval_steps: int = 500
    logging_steps: int = 500
    save_strategy: str = "no"
    save_steps: int = 500
    save_total_limit: Optional[int] = None
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    seed: int = 42
    report_to: str = "wandb"
    logging_dir: Path = Path("logs")
    run_name: Optional[str] = None
    wandb_project: Optional[str] = "DemPO"
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
        self._ensure_length_column()

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
            # Per-batch self-normalization: use weighted *mean* instead of weighted sum so update
            # magnitudes remain comparable across training variants.
            denom = weight_tensor.sum()
            count = torch.tensor(
                weight_tensor.numel(),
                device=weight_tensor.device,
                dtype=weight_tensor.dtype,
            )
            if getattr(self, "accelerator", None) is not None:
                denom = self.accelerator.reduce(denom, reduction="sum")
                count = self.accelerator.reduce(count, reduction="sum")
            mean_w = denom / count.clamp_min(1.0)
            weight_tensor = weight_tensor / mean_w.clamp_min(1e-12)
            losses = losses * weight_tensor
        return losses, chosen_rewards, rejected_rewards

    def _ensure_length_column(self) -> None:
        """
        Transformers' LengthGroupedSampler expects either:
        - a dataset column named `args.length_column_name`, or
        - examples with an `input_ids` field (to infer lengths).

        TRL's DPO preprocessing produces `prompt_input_ids`, `chosen_input_ids`,
        and `rejected_input_ids` instead of `input_ids`, so we attach a `length`
        column *after* TRL tokenization to make `group_by_length=True` work.
        """

        train_sampling_strategy = getattr(self.args, "train_sampling_strategy", None)
        if not (
            getattr(self.args, "group_by_length", False)
            or train_sampling_strategy == "group_by_length"
        ):
            return

        length_col = getattr(self.args, "length_column_name", "length")

        def _add_length(ds, name: str):
            if ds is None:
                return ds
            if not hasattr(ds, "column_names") or not hasattr(ds, "add_column"):
                return ds
            if length_col in ds.column_names:
                return ds

            cols = set(ds.column_names)
            if {"prompt_input_ids", "chosen_input_ids", "rejected_input_ids"}.issubset(cols):
                prompts = ds["prompt_input_ids"]
                chosen = ds["chosen_input_ids"]
                rejected = ds["rejected_input_ids"]
                lengths = [
                    int(len(p) + max(len(c), len(r)))
                    for p, c, r in zip(prompts, chosen, rejected)
                ]
            elif "input_ids" in cols:
                lengths = [int(len(ids)) for ids in ds["input_ids"]]
            else:
                print(
                    f"Warning: group_by_length=True but couldn't infer lengths for {name}; "
                    f"missing expected token columns: {ds.column_names}"
                )
                return ds

            try:
                return ds.add_column(length_col, lengths)
            except Exception as exc:
                print(f"Warning: failed to add '{length_col}' column for {name}: {exc}")
                return ds

        self.train_dataset = _add_length(getattr(self, "train_dataset", None), "train_dataset")
        eval_ds = getattr(self, "eval_dataset", None)
        if isinstance(eval_ds, dict):
            self.eval_dataset = {k: _add_length(v, f"eval_dataset[{k}]") for k, v in eval_ds.items()}
        else:
            self.eval_dataset = _add_length(eval_ds, "eval_dataset")


def load_tokenizer(model_id: str, token: Optional[str]):
    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    if not getattr(tok, "chat_template", None):
        tok.chat_template = DEFAULT_CHAT_TEMPLATE
    return tok


def load_model(
    model_id: str,
    token: Optional[str],
    device_map: Optional[str] = "auto",
    attn_implementation: Optional[str] = "flash_attention_2",
):
    kwargs = dict(
        token=token,
        dtype=torch.bfloat16,
        device_map=device_map,
    )
    if attn_implementation and attn_implementation.lower() not in {"none", "null"}:
        kwargs["attn_implementation"] = attn_implementation
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def build_datasets(
    path: Path,
    eval_ratio: float,
    seed: int,
):
    if path.exists():
        with path.open("rb") as f:
            header = f.read(200)
        if b"git-lfs" in header and b"version https://git-lfs.github.com/spec/v1" in header:
            raise RuntimeError(
                f"{path} looks like a Git LFS pointer file, not a real JSONL dataset. "
                "This usually means the dataset artifacts were not generated on this machine. "
                "Re-run `python scripts/prepare_data.py ...` to create the JSONL, or run `git lfs pull` "
                "if you intentionally tracked artifacts with Git LFS."
            )
    dataset = datasets.load_dataset("json", data_files=str(path))["train"]
    dataset = dataset.shuffle(seed=seed)
    if eval_ratio and eval_ratio > 0:
        split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
        return split["train"], split["test"]
    return dataset, None


def build_compatible_dpo_config_kwargs(dpo_kwargs: dict) -> tuple[dict, dict, list[str]]:
    """
    Filter/rename kwargs based on the installed TRL DPOConfig signature.

    TRL/transformers versions differ on which TrainingArguments fields DPOConfig
    accepts in __init__. Recent Transformers docs use
    `train_sampling_strategy="group_by_length"`, while older versions expose
    `group_by_length=True`. Normalize between the two and apply unsupported
    fields post-init when possible.
    """

    supported = set(inspect.signature(DPOConfig.__init__).parameters.keys())
    kwargs = dict(dpo_kwargs)
    post_init_attrs: dict = {}

    # Known alias across transformers/trl versions.
    if "eval_strategy" not in supported and "evaluation_strategy" in supported and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    elif "evaluation_strategy" not in supported and "eval_strategy" in supported and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")

    if "train_sampling_strategy" in supported:
        if kwargs.pop("group_by_length", False):
            kwargs["train_sampling_strategy"] = "group_by_length"
    elif "group_by_length" in supported:
        if kwargs.get("train_sampling_strategy") == "group_by_length":
            kwargs["group_by_length"] = True
        kwargs.pop("train_sampling_strategy", None)

    unsupported: list[str] = []
    filtered: dict = {}
    for key, value in kwargs.items():
        if key in supported:
            filtered[key] = value
        else:
            unsupported.append(key)
            post_init_attrs[key] = value
    return filtered, post_init_attrs, sorted(unsupported)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Llama-3.1-8B (base) with DPO on PRISM-prepared data.")
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL with prompt/chosen/rejected (+ optional weight).")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/llama3.1-8b-dpo"))
    parser.add_argument("--model-id", default="meta-llama/Llama-3.1-8B")
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map for model loading ('auto', 'balanced', or 'none'). "
        "When running distributed, 'auto' is treated as 'none' so FSDP/DDP can shard.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help="Attention implementation passed to from_pretrained (e.g. flash_attention_2, sdpa, eager, none).",
    )
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=1,
        help="Per-device batch size. Used for both training and evaluation.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        dest="per_device_batch_size",
        type=int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument(
        "--num-train-steps",
        type=int,
        default=-1,
        help="If > 0, train for exactly this many optimizer steps (overrides --num-train-epochs).",
    )
    parser.add_argument(
        "--max-steps",
        dest="num_train_steps",
        type=int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-ratio", type=float, default=0.02)
    parser.add_argument(
        "--eval-strategy",
        choices=["no", "steps", "epoch"],
        default="steps",
        help="Evaluation strategy (ignored if eval_ratio=0).",
    )
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=500, help="Log metrics every N steps.")
    parser.add_argument(
        "--save-strategy",
        choices=["no", "steps", "epoch"],
        default="no",
        help="Checkpoint saving strategy (defaults to only saving the final model).",
    )
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps.")
    parser.add_argument("--save-total-limit", type=int, default=None, help="Max number of checkpoints to keep.")
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional max sequence length. If omitted, uses model max_position_embeddings.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=None,
        help="Optional max prompt length. If omitted, uses max_length.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", default="wandb", help="Logging backend (wandb, tensorboard, or none).")
    parser.add_argument("--logging-dir", type=Path, default=Path("logs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "DemPO"))
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=None)
    return parser.parse_args()


def resolve_device_map(requested: str | None, distributed: bool) -> Optional[str]:
    if requested is None:
        return None
    lowered = requested.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered == "auto" and distributed:
        return None
    return requested


def main() -> None:
    args = parse_args()
    if "instruct" in args.model_id.lower():
        print(
            "Warning: model-id looks like an instruction-tuned checkpoint. "
            "This paper's experiments use the base model; results may not be comparable."
        )
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1 or os.environ.get("LOCAL_RANK") is not None
    device_map = resolve_device_map(args.device_map, distributed)
    if distributed and args.device_map.lower() == "auto":
        print("Distributed run detected; disabling device_map so FSDP/DDP can manage sharding.")
    if args.run_name is None:
        dataset_tag = args.dataset.stem
        model_tag = args.model_id.split("/")[-1]
        ts = time.strftime("%Y%m%d-%H%M%S")
        args.run_name = f"dpo-{model_tag}-{dataset_tag}-{ts}"
    cfg = TrainConfig(
        model_id=args.model_id,
        output_dir=args.output_dir,
        dataset_path=args.dataset,
        hf_token=args.hf_token,
        device_map=device_map,
        attn_implementation=args.attn_implementation,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        num_train_steps=args.num_train_steps,
        beta=args.beta,
        weight_decay=args.weight_decay,
        eval_ratio=args.eval_ratio,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        seed=args.seed,
        report_to=args.report_to,
        logging_dir=args.logging_dir,
        run_name=args.run_name,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
    )
    dataloader_num_workers = args.dataloader_num_workers
    dataloader_prefetch_factor = args.dataloader_prefetch_factor

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

    print(
        "Training config: "
        f"model_id={cfg.model_id}, "
        f"device_map={cfg.device_map}, "
        f"attn_implementation={cfg.attn_implementation}, "
        f"per_device_batch_size={cfg.per_device_batch_size}, "
        f"gradient_accumulation_steps={cfg.gradient_accumulation_steps}, "
        f"report_to={cfg.report_to}"
    )

    tokenizer = load_tokenizer(cfg.model_id, cfg.hf_token)
    model = load_model(cfg.model_id, cfg.hf_token, cfg.device_map, cfg.attn_implementation)
    ref_model = load_model(cfg.model_id, cfg.hf_token, cfg.device_map, cfg.attn_implementation)

    model_max = getattr(model.config, "max_position_embeddings", None)
    if cfg.max_length is None and isinstance(model_max, int) and model_max > 0:
        cfg.max_length = model_max
    if cfg.max_prompt_length is None and cfg.max_length is not None:
        cfg.max_prompt_length = cfg.max_length

    train_ds, eval_ds = build_datasets(cfg.dataset_path, cfg.eval_ratio, cfg.seed)
    print(f"Loaded dataset: {len(train_ds)} train rows" + (f", {len(eval_ds)} eval rows" if eval_ds else ""))
    if cfg.num_train_steps and cfg.num_train_steps > 0:
        print(
            f"Using num_train_steps={cfg.num_train_steps}; "
            "--num-train-epochs will be ignored by Trainer."
        )
    report_to = [] if cfg.report_to == "none" else [cfg.report_to]
    eval_strategy = cfg.eval_strategy if eval_ds is not None else "no"
    dpo_kwargs = dict(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=cfg.logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=cfg.eval_steps,
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=True,
        weight_decay=cfg.weight_decay,
        seed=cfg.seed,
        report_to=report_to,
        logging_dir=str(cfg.logging_dir),
        run_name=cfg.run_name,
        beta=cfg.beta,
        dataloader_num_workers=dataloader_num_workers,
        train_sampling_strategy="group_by_length",
        length_column_name="length",
        remove_unused_columns=False,
        save_strategy=cfg.save_strategy,
    )
    if cfg.num_train_steps and cfg.num_train_steps > 0:
        dpo_kwargs["max_steps"] = cfg.num_train_steps
    if cfg.max_length is not None:
        dpo_kwargs["max_length"] = cfg.max_length
    if cfg.max_prompt_length is not None:
        dpo_kwargs["max_prompt_length"] = cfg.max_prompt_length
    if cfg.save_strategy == "steps":
        dpo_kwargs["save_steps"] = cfg.save_steps
    if cfg.save_total_limit is not None:
        dpo_kwargs["save_total_limit"] = cfg.save_total_limit
    if dataloader_prefetch_factor is not None:
        dpo_kwargs["dataloader_prefetch_factor"] = dataloader_prefetch_factor
    filtered_dpo_kwargs, post_init_attrs, unsupported = build_compatible_dpo_config_kwargs(dpo_kwargs)
    if unsupported:
        print(
            "Installed TRL DPOConfig does not accept these constructor args; "
            f"applying them after init when possible: {', '.join(unsupported)}"
        )
    training_args = DPOConfig(**filtered_dpo_kwargs)
    for key, value in post_init_attrs.items():
        try:
            setattr(training_args, key, value)
        except Exception as exc:
            print(f"Warning: failed to set post-init DPOConfig attribute {key}={value!r}: {exc}")

    data_collator = WeightedDataCollatorForPreference(pad_token_id=tokenizer.pad_token_id)
    trainer = WeightedDPOTrainer(
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

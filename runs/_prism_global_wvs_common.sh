#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PYTHON="${PYTHON:-python}"

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B}"
SHORT="${SHORT:-llama3.1-8b-global-wvs}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-3538}"

EVAL_RATIO="${EVAL_RATIO:-0.0}"
EVAL_STRATEGY="${EVAL_STRATEGY:-no}"
EVAL_STEPS="${EVAL_STEPS:-500}"
SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-500}"

MAX_LENGTH="${MAX_LENGTH:-}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-}"
TRUNCATION_MODE="${TRUNCATION_MODE:-keep_end}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-12}"
PRECOMPUTE_REF_LOG_PROBS="${PRECOMPUTE_REF_LOG_PROBS:-0}"
PRECOMPUTE_REF_BATCH_SIZE="${PRECOMPUTE_REF_BATCH_SIZE:-}"

SEED="${SEED:-42}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-DemPO}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-prism-global-wvs-8b}"

HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

build_common_train_cmd() {
  local -n _out=$1
  _out=(
    "$PYTHON" scripts/train_dpo.py
    --model-id "$MODEL_ID"
    --per-device-batch-size "$PER_DEVICE_BATCH_SIZE"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning-rate "$LEARNING_RATE"
    --eval-ratio "$EVAL_RATIO"
    --eval-strategy "$EVAL_STRATEGY"
    --eval-steps "$EVAL_STEPS"
    --save-strategy "$SAVE_STRATEGY"
    --save-steps "$SAVE_STEPS"
    --logging-steps "$LOGGING_STEPS"
    --seed "$SEED"
    --report-to "$REPORT_TO"
    --wandb-project "$WANDB_PROJECT"
    --wandb-group "$WANDB_GROUP"
    --dataset-num-proc "$DATASET_NUM_PROC"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
    --dataloader-prefetch-factor "$DATALOADER_PREFETCH_FACTOR"
    --attn-implementation "$ATTN_IMPLEMENTATION"
    --device-map "$DEVICE_MAP"
    --truncation-mode "$TRUNCATION_MODE"
    --num-train-steps "$NUM_TRAIN_STEPS"
  )
  if [[ -n "$HF_TOKEN" ]]; then
    _out+=(--hf-token "$HF_TOKEN")
  fi
  if [[ -n "$WANDB_ENTITY" ]]; then
    _out+=(--wandb-entity "$WANDB_ENTITY")
  fi
  if [[ -n "$MAX_LENGTH" ]]; then
    _out+=(--max-length "$MAX_LENGTH")
  fi
  if [[ -n "$MAX_PROMPT_LENGTH" ]]; then
    _out+=(--max-prompt-length "$MAX_PROMPT_LENGTH")
  fi
  if [[ "$PRECOMPUTE_REF_LOG_PROBS" == "1" ]]; then
    _out+=(--precompute-ref-log-probs)
  fi
  if [[ -n "$PRECOMPUTE_REF_BATCH_SIZE" ]]; then
    _out+=(--precompute-ref-batch-size "$PRECOMPUTE_REF_BATCH_SIZE")
  fi
}

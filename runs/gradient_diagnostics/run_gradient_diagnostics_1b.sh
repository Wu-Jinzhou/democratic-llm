#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-1B}"
SHORT="${SHORT:-llama3.2-1b}"

SOFT_DATASET="${SOFT_DATASET:-artifacts/data/soft_panel.jsonl}"
HARD_DATASET="${HARD_DATASET:-artifacts/data/hard_panel.jsonl}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-12}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-300}"
SEED="${SEED:-42}"

TRAIN_SAMPLING_STRATEGY="${TRAIN_SAMPLING_STRATEGY:-random}"
REPORT_TO="${REPORT_TO:-none}"
LOGGING_STEPS="${LOGGING_STEPS:-25}"
SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
EVAL_RATIO="${EVAL_RATIO:-0}"

GRADIENT_WINDOW_START="${GRADIENT_WINDOW_START:-51}"
GRADIENT_WINDOW_SIZE="${GRADIENT_WINDOW_SIZE:-50}"
GRADIENT_SKETCH_SIZE="${GRADIENT_SKETCH_SIZE:-4096}"
GRADIENT_DIAGNOSTICS_STEPS="${GRADIENT_DIAGNOSTICS_STEPS:-300}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"

OUTDIR="${OUTDIR:-artifacts/evaluations/diagnostics/gradient_hard_vs_soft_1b}"
SOFT_RUN_DIR="${OUTDIR}/soft"
HARD_RUN_DIR="${OUTDIR}/hard"
COMPARE_DIR="${OUTDIR}/comparison"

SOFT_CKPT="${SOFT_CKPT:-checkpoints/${SHORT}-soft-panel-gradient-diagnostics}"
HARD_CKPT="${HARD_CKPT:-checkpoints/${SHORT}-hard-panel-gradient-diagnostics}"

FORCE_RERUN="${FORCE_RERUN:-0}"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

mkdir -p "$SOFT_RUN_DIR" "$HARD_RUN_DIR" "$COMPARE_DIR"

run_condition() {
  local tag="$1"
  local dataset="$2"
  local outdir="$3"
  local diag_dir="$4"

  if [[ "$FORCE_RERUN" != "1" && -f "$diag_dir/metadata.json" && -f "$diag_dir/gradient_summaries.csv" ]]; then
    log "Skipping ${tag}; diagnostics already exist at ${diag_dir}"
    return
  fi

  log "Training ${tag} diagnostics run"
  log "  dataset=${dataset}"
  log "  output_dir=${outdir}"
  log "  diagnostics_dir=${diag_dir}"

  "$PYTHON" -u scripts/train_dpo.py \
    --dataset "$dataset" \
    --model-id "$MODEL_ID" \
    --output-dir "$outdir" \
    --per-device-batch-size "$PER_DEVICE_BATCH_SIZE" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning-rate "$LEARNING_RATE" \
    --dataset-num-proc "$DATASET_NUM_PROC" \
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
    --num-train-steps "$NUM_TRAIN_STEPS" \
    --eval-ratio "$EVAL_RATIO" \
    --save-strategy "$SAVE_STRATEGY" \
    --logging-steps "$LOGGING_STEPS" \
    --seed "$SEED" \
    --report-to "$REPORT_TO" \
    --train-sampling-strategy "$TRAIN_SAMPLING_STRATEGY" \
    --gradient-diagnostics-dir "$diag_dir" \
    --gradient-diagnostics-max-steps "$GRADIENT_DIAGNOSTICS_STEPS" \
    --gradient-diagnostics-window-start "$GRADIENT_WINDOW_START" \
    --gradient-diagnostics-window-size "$GRADIENT_WINDOW_SIZE" \
    --gradient-diagnostics-sketch-size "$GRADIENT_SKETCH_SIZE" \
    --gradient-diagnostics-seed "$SEED"
}

log "Gradient diagnostics configuration:"
log "  MODEL_ID=$MODEL_ID"
log "  PER_DEVICE_BATCH_SIZE=$PER_DEVICE_BATCH_SIZE"
log "  GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"
log "  NUM_TRAIN_STEPS=$NUM_TRAIN_STEPS"
log "  TRAIN_SAMPLING_STRATEGY=$TRAIN_SAMPLING_STRATEGY"
log "  GRADIENT_WINDOW_START=$GRADIENT_WINDOW_START"
log "  GRADIENT_WINDOW_SIZE=$GRADIENT_WINDOW_SIZE"
log "  GRADIENT_SKETCH_SIZE=$GRADIENT_SKETCH_SIZE"
log "  OUTDIR=$OUTDIR"

run_condition "soft-panel" "$SOFT_DATASET" "$SOFT_CKPT" "$SOFT_RUN_DIR"
run_condition "hard-panel" "$HARD_DATASET" "$HARD_CKPT" "$HARD_RUN_DIR"

log "Comparing hard vs soft gradient diagnostics"
"$PYTHON" -u scripts/analysis/gradient_diagnostics.py \
  --hard-dir "$HARD_RUN_DIR" \
  --soft-dir "$SOFT_RUN_DIR" \
  --output-dir "$COMPARE_DIR" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --bootstrap-seed "$SEED"

log "Done. Outputs written under $OUTDIR"

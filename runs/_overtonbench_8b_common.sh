#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PYTHON="${PYTHON:-python}"

BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B}"
INCLUDE_BASE="${INCLUDE_BASE:-0}"
MODEL_SET="${MODEL_SET:-all}"

PRISM_FULL_MODEL="${PRISM_FULL_MODEL:-checkpoints/llama3.1-8b-full-prism}"
PRISM_SOFT_MODEL="${PRISM_SOFT_MODEL:-checkpoints/llama3.1-8b-soft-panel}"
PRISM_HARD_MODEL="${PRISM_HARD_MODEL:-checkpoints/llama3.1-8b-hard-panel}"
PRISM_US_REP_MODEL="${PRISM_US_REP_MODEL:-checkpoints/llama3.1-8b-us-rep}"

COMMUNITY_FULL_GLOBAL_MODEL="${COMMUNITY_FULL_GLOBAL_MODEL:-checkpoints/community-alignment/llama3.1-8b-full-global}"
COMMUNITY_FULL_US_MODEL="${COMMUNITY_FULL_US_MODEL:-checkpoints/community-alignment/llama3.1-8b-full-us}"
COMMUNITY_SOFT_US_MODEL="${COMMUNITY_SOFT_US_MODEL:-checkpoints/community-alignment/llama3.1-8b-soft-us}"
COMMUNITY_HARD_US_MODEL="${COMMUNITY_HARD_US_MODEL:-checkpoints/community-alignment/llama3.1-8b-hard-us}"

OVERTON_QUESTIONS_CSV="${OVERTON_QUESTIONS_CSV:-/overtonbench/meta/questions.csv}"
OVERTON_BENCHMARK_CSV="${OVERTON_BENCHMARK_CSV:-/overtonbench/data/prolific_with_clusters_kmeans_merged_public.csv}"
OVERTON_SOURCE="${OVERTON_SOURCE:-full}"

GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"

JUDGE_MODEL="${JUDGE_MODEL:-gemini-2.5-pro}"
JUDGE_PROMPT="${JUDGE_PROMPT:-fs+fr}"
JUDGE_WORKERS="${JUDGE_WORKERS:-64}"

HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

EVAL_DIR="${EVAL_DIR:-artifacts/evaluations/overtonbench/llama3.1-8b}"
RESPONSES_DIR="${RESPONSES_DIR:-$EVAL_DIR/responses}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-$EVAL_DIR/predictions_gemini25pro_fsfr}"
SUMMARY_JSON="${SUMMARY_JSON:-$EVAL_DIR/summary_gemini25pro_fsfr.json}"
SUMMARY_CSV="${SUMMARY_CSV:-$EVAL_DIR/summary_gemini25pro_fsfr.csv}"

case "$MODEL_SET" in
  prism)
    MODEL_LIST=(
      "$PRISM_FULL_MODEL"
      "$PRISM_SOFT_MODEL"
      "$PRISM_HARD_MODEL"
      "$PRISM_US_REP_MODEL"
    )
    ;;
  community)
    MODEL_LIST=(
      "$COMMUNITY_FULL_GLOBAL_MODEL"
      "$COMMUNITY_FULL_US_MODEL"
      "$COMMUNITY_SOFT_US_MODEL"
      "$COMMUNITY_HARD_US_MODEL"
    )
    ;;
  all)
    MODEL_LIST=(
      "$PRISM_FULL_MODEL"
      "$PRISM_SOFT_MODEL"
      "$PRISM_HARD_MODEL"
      "$PRISM_US_REP_MODEL"
      "$COMMUNITY_FULL_GLOBAL_MODEL"
      "$COMMUNITY_FULL_US_MODEL"
      "$COMMUNITY_SOFT_US_MODEL"
      "$COMMUNITY_HARD_US_MODEL"
    )
    ;;
  *)
    echo "Invalid MODEL_SET=$MODEL_SET. Expected one of: all, prism, community." >&2
    exit 1
    ;;
esac

if [[ "$INCLUDE_BASE" == "1" ]]; then
  MODEL_LIST=("$BASE_MODEL" "${MODEL_LIST[@]}")
fi

mkdir -p "$RESPONSES_DIR" "$PREDICTIONS_DIR"

HF_ARGS=()
if [[ -n "$HF_TOKEN" ]]; then
  HF_ARGS+=(--hf-token "$HF_TOKEN")
fi

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}
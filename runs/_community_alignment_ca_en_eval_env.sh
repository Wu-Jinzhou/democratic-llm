#!/usr/bin/env bash
set -euo pipefail

export FULL_EN_GLOBAL_MODEL="${FULL_EN_GLOBAL_MODEL:-${FULL_GLOBAL_MODEL:-checkpoints/community-alignment/llama3.1-8b-full-en-global}}"
export FULL_US_MODEL="${FULL_US_MODEL:-checkpoints/community-alignment/llama3.1-8b-full-us}"
export SOFT_MODEL="${SOFT_MODEL:-checkpoints/community-alignment/llama3.1-8b-soft-en}"
export HARD_MODEL="${HARD_MODEL:-checkpoints/community-alignment/llama3.1-8b-hard-en}"

export EVAL_DIR="${EVAL_DIR:-artifacts/evaluations/community-alignment-en/llama3.1-8b/no_system_prompt}"

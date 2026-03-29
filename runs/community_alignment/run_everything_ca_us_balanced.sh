#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_community_alignment_common.sh"

RUN_PREP="${RUN_PREP:-0}"
RUN_GENERATE="${RUN_GENERATE:-1}"
RUN_JUDGE="${RUN_JUDGE:-1}"
COMMUNITY_ALIGNMENT_NUM_TRAIN_STEPS="${COMMUNITY_ALIGNMENT_NUM_TRAIN_STEPS:-7014}"
export COMMUNITY_ALIGNMENT_NUM_TRAIN_STEPS

log "Community Alignment US-balanced end-to-end run configuration:"
log "  RUN_PREP=$RUN_PREP"
log "  RUN_GENERATE=$RUN_GENERATE"
log "  RUN_JUDGE=$RUN_JUDGE"
log "  COMMUNITY_ALIGNMENT_NUM_TRAIN_STEPS=$COMMUNITY_ALIGNMENT_NUM_TRAIN_STEPS"

if [[ "$RUN_PREP" == "1" ]]; then
  "$SCRIPT_DIR/prepare_community_alignment_us_balanced.sh"
fi

"$SCRIPT_DIR/train_community_alignment_8b_us_balanced_subset.sh"
"$SCRIPT_DIR/train_community_alignment_8b_soft_us_k350.sh"
"$SCRIPT_DIR/train_community_alignment_8b_hard_us_k350.sh"

if [[ "$RUN_GENERATE" == "1" ]]; then
  "$SCRIPT_DIR/generate_ca_us_balanced_responses.sh"
fi

if [[ "$RUN_JUDGE" == "1" ]]; then
  "$SCRIPT_DIR/judge_ca_us_balanced.sh"
fi

log "Completed Community Alignment US-balanced pipeline."

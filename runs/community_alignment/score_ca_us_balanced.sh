#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_community_alignment_ca_us_balanced_eval_env.sh"

export PRIMARY_BASELINE_MODEL="${PRIMARY_BASELINE_MODEL:-$FULL_US_MODEL}"
export SECONDARY_BASELINE_MODEL="${SECONDARY_BASELINE_MODEL:-$BASE_MODEL}"

exec "$SCRIPT_DIR/score_community_alignment_8b.sh"

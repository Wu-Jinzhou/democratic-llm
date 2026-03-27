#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

run_train "$SOFT_CLIPPED_DATASET" "$SOFT_CLIPPED_MODEL" "Training PRISM soft-panel clipped-pi_i model."

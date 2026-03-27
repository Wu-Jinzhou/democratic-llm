#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

run_train "$SOFT_SQUARE_DATASET" "$SOFT_SQUARE_MODEL" "Training PRISM soft-panel pi_i^2 model."

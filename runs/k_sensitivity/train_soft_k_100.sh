#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

run_train "$SOFT_K100_DATASET" "$SOFT_K100_MODEL" "Training PRISM soft-panel control model (k=100)."

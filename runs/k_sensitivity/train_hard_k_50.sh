#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

run_train "$HARD_K50_DATASET" "$HARD_K50_MODEL" "Training PRISM hard-panel control model (k=50)."

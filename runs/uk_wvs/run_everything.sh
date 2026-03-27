#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_PREP="${RUN_PREP:-0}"
RUN_EVAL="${RUN_EVAL:-1}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

log "UK PRISM + WVS run configuration:"
log "  RUN_PREP=$RUN_PREP"
log "  RUN_EVAL=$RUN_EVAL"

if [[ "$RUN_PREP" == "1" ]]; then
  "$SCRIPT_DIR/prepare_prism_uk_wvs.sh"
else
  log "Skipping dataset preparation because RUN_PREP=$RUN_PREP."
fi

"$SCRIPT_DIR/train_prism_uk_8b_uk_rep.sh"
"$SCRIPT_DIR/train_prism_uk_8b_soft.sh"
"$SCRIPT_DIR/train_prism_uk_8b_hard.sh"

if [[ "$RUN_EVAL" == "1" ]]; then
  "$SCRIPT_DIR/evaluate_wvs_8b.sh"
else
  log "Skipping WVS evaluation because RUN_EVAL=$RUN_EVAL."
fi

log "UK PRISM + WVS run completed."

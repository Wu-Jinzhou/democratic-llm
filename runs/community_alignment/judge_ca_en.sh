#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_community_alignment_ca_en_eval_env.sh"

exec "$SCRIPT_DIR/judge_community_alignment_8b.sh"

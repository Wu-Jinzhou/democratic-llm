#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/generate_community_alignment_8b_responses.sh"
"$SCRIPT_DIR/judge_community_alignment_8b.sh"

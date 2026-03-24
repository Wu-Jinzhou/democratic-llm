#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/generate_overtonbench_8b_responses.sh"
"$SCRIPT_DIR/judge_overtonbench_8b.sh"
"$SCRIPT_DIR/summarize_overtonbench_8b.sh"

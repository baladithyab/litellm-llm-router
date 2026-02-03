#!/usr/bin/env bash
# Secret Detection Wrapper for Lefthook
# Delegates to the Python implementation for OS-agnostic behavior
#
# Usage:
#   ./scripts/detect_secrets.sh [files...]        # Scan specific files
#   ./scripts/detect_secrets.sh --full-scan       # Scan entire repo (pre-push)

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use uv if available, otherwise fall back to python
run_python() {
    if command -v uv &>/dev/null; then
        uv run python "$@"
    elif command -v python3 &>/dev/null; then
        python3 "$@"
    else
        python "$@"
    fi
}

run_python "$SCRIPT_DIR/detect_secrets.py" "$@"

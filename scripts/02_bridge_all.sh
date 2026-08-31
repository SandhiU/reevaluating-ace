#!/bin/bash
# ============================================================
# Stage 2: Bridge all CTRAIN checkpoints to ACE format
# Usage: bash 02_bridge_all.sh [--dry-run]
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/00_config.sh"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

echo "=== Stage 2: Bridge checkpoints ==="
echo "  Source: $RUNS_DIR"
echo "  Dest:   $CONVERTED_DIR"

# Ensure output dir exists
mkdir -p "$CONVERTED_DIR"

cd "$SCRIPTS_DIR"
if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [dry run] python convert_format.py --all --runs-dir $RUNS_DIR --output-dir $CONVERTED_DIR"
else
    python convert_format.py --all --runs-dir "$RUNS_DIR" --output-dir "$CONVERTED_DIR"
fi

echo ""
echo "=== Bridge complete. Check converted files: ==="
ls -la "$CONVERTED_DIR"/*.pt 2>/dev/null || echo "  (no .pt files found)"

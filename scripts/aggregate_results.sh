#!/bin/bash
# ============================================================
# Aggregate ACE evaluation results into a CSV
# Usage: bash aggregate_results.sh [version] [extra_args...]
#   version = v1, v2, v3, ... or "all" (default: all)
#
# Examples:
#   bash aggregate_results.sh v3           # only v3 eval results
#   bash aggregate_results.sh all          # everything
#   bash aggregate_results.sh v3 --label sabr_2_255
#
# Output: ~/research/results/agg_<version>.csv (or agg_all.csv)
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/00_config.sh"

VERSION="${1:-all}"
shift 2>/dev/null || true  # consume version arg

RESULTS_DIR="$ACE_DIR/models_new"
mkdir -p "$RESEARCH_DIR/results"

if [[ "$VERSION" == "all" ]]; then
    OUTPUT="$RESEARCH_DIR/results/agg_all.csv"
    VERSION_FLAG=""
else
    OUTPUT="$RESEARCH_DIR/results/agg_${VERSION}.csv"
    VERSION_FLAG="--version $VERSION"
fi

echo "=== Aggregating ACE results ==="
echo "  Source:  $RESULTS_DIR"
echo "  Output:  $OUTPUT"
echo "  Version: $VERSION"

cd "$SCRIPTS_DIR"
python aggregate_ace_results.py \
    --results-dir "$RESULTS_DIR" \
    --output "$OUTPUT" \
    $VERSION_FLAG \
    "$@"

echo ""
echo "=== Done. Preview: ==="
if command -v column &>/dev/null; then
    head -20 "$OUTPUT" | column -t -s,
else
    head -20 "$OUTPUT"
fi

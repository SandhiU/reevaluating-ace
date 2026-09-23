#!/usr/bin/env bash
# ============================================================
# pipeline_status.sh -- what each stage has produced for a version
# Usage: bash pipeline_status.sh [ver]   (default: $DEFAULT_VER)
# ============================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

VER="${1:-$DEFAULT_VER}"
MODELS="$ACE_DIR/models_new/cifar10"

core_ok()  { [[ -n "$(find "$MODELS/core_adv_$1_$VER" -name 'best_model.pt' -o -name 'final_model_state.pt' 2>/dev/null | head -1 || true)" ]]; }
branch_ok(){ [[ -f "$(branch_pt "$1" "$2" "$VER")" ]]; }
labels_ok(){ [[ -f "$(label_csv "$1" "$2" "$VER")" ]]; }
sel_ok()   { [[ -n "$(find_selector "$(sel_name "$1" "$2" "$VER" "$3")")" ]]; }

echo "pipeline status  ver=$VER"
echo ""
printf "  %-10s %-7s %-6s %-8s %-8s %-6s\n" METHOD EPS CORE BRANCH LABELS SELECTOR
for m in ibp sabr mtl_ibp crown_ibp; do
    for e in 2_255 8_255; do
        printf "  %-10s %-7s %-6s %-8s %-8s %-6s\n" "$m" "$e" \
            "$(core_ok "$e" && echo ok || echo -)" \
            "$(branch_ok "$m" "$e" && echo ok || echo -)" \
            "$(labels_ok "$m" "$e" && echo ok || echo -)" \
            "$(sel_ok "$m" "$e" sel && echo ok || echo -)"
    done
done
echo ""
echo "evals: $MODELS/*_eval_${VER}/   (a done run has cert_log.csv)"

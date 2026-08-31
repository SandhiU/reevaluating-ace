#!/bin/bash
# ============================================================
# Stage 3: Train selection mechanisms (SelNet + Entropy)
# Usage: bash 03_train_selector_all.sh [version] [--sel|--ent|--both] [converted_dir]
#   version       = v1 (default), v2, v3, ...
#   --sel         = SelNet only
#   --ent         = Entropy only
#   --both        = both (default)
#   converted_dir = where to find *_branch.pt files (default: ~/research/converted)
#
# The converted directory is NOT versioned — branch files are named
# ibp_2_255_branch.pt, sabr_8_255_branch.pt, etc. regardless of which
# training run produced them. If you re-train branches, re-run stage 2
# first to update the converted directory.
#
# Examples:
#   bash 03_train_selector_all.sh v1              # both gates, default converted dir
#   bash 03_train_selector_all.sh v1 --sel        # SelNet only
#   bash 03_train_selector_all.sh v1 --ent /path/to/converted  # custom dir
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/00_config.sh"

VERSION="${1:-v1}"
GATE_FILTER="${2:---both}"
CONVERTED_DIR="${3:-$RESEARCH_DIR/converted}"

# exp-id scheme: base 100
# sel: 100 + method_idx*2 + eps_idx
# ent: 110 + method_idx*2 + eps_idx
# method_idx: ibp=0, sabr=1, mtl=2, crown=3
# eps_idx: 2/255=0, 8/255=1
declare -A METHOD_IDX=( [ibp]=0 [sabr]=1 [mtl_ibp]=2 [crown_ibp]=3 )

METHODS=(ibp sabr mtl_ibp crown_ibp)
EPS_LABELS=("2_255" "8_255")
EPS_VALUES=("$EPS_2" "$EPS_8")

do_sel() {
    echo "  --- SelNet selectors ---"
    for method in "${METHODS[@]}"; do
        for i in 0 1; do
            eps_label="${EPS_LABELS[$i]}"
            eps="${EPS_VALUES[$i]}"
            branch="$CONVERTED_DIR/${method}_${eps_label}_branch.pt"
            exp_id=$(( 100 + METHOD_IDX[$method]*2 + i ))
            exp_name="${method}_${eps_label}_${VERSION}"

            if [[ ! -f "$branch" ]]; then
                echo "  SKIP $method $eps_label: branch not found ($branch)"
                continue
            fi

            echo "  submitting: sel_${method}_${eps_label} (exp-id=$exp_id)"
            cd "$SCRIPTS_DIR"
            sbatch --job-name="sel_${method}_${eps_label}_${VERSION}" \
                train_selector.slurm \
                --branch "$branch" \
                --gate-type net \
                --eps "$eps" \
                --exp-id "$exp_id" \
                --exp-name "$exp_name"
        done
    done
}

do_ent() {
    echo "  --- Entropy selectors ---"
    for method in "${METHODS[@]}"; do
        for i in 0 1; do
            eps_label="${EPS_LABELS[$i]}"
            eps="${EPS_VALUES[$i]}"
            branch="$CONVERTED_DIR/${method}_${eps_label}_branch.pt"
            exp_id=$(( 110 + METHOD_IDX[$method]*2 + i ))
            exp_name="${method}_${eps_label}_ent_${VERSION}"

            if [[ ! -f "$branch" ]]; then
                echo "  SKIP $method $eps_label: branch not found ($branch)"
                continue
            fi

            echo "  submitting: ent_${method}_${eps_label} (exp-id=$exp_id)"
            cd "$SCRIPTS_DIR"
            sbatch --job-name="ent_${method}_${eps_label}_${VERSION}" \
                train_selector.slurm \
                --branch "$branch" \
                --gate-type entropy \
                --eps "$eps" \
                --exp-id "$exp_id" \
                --exp-name "$exp_name"
        done
    done
}

echo "=== Stage 3: Train selectors ($VERSION) ==="
echo "  Converted dir: $CONVERTED_DIR"

case "$GATE_FILTER" in
    --sel)  do_sel ;;
    --ent)  do_ent ;;
    --both) do_sel; echo ""; do_ent ;;
    *)      echo "Unknown filter: $GATE_FILTER (use --sel, --ent, or --both)"; exit 1 ;;
esac

echo ""
echo "=== All stage 3 jobs submitted. Check with: squeue -u \$USER ==="

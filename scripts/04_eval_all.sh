#!/bin/bash
# ============================================================
# Stage 4: Evaluate all ACE compositions (τ-sweep)
# Usage: bash 04_eval_all.sh [version] [--sel|--ent|--both]
#   version = v1 (default), v2, v3, ...
#   --sel   = SelNet only
#   --ent   = Entropy only
#   --both  = both (default)
#
# Auto-discovers ACE models from stage 3 by searching models_new/
# for matching exp-name patterns. No hardcoded exp-ids needed.
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/00_config.sh"

VERSION="${1:-v1}"
GATE_FILTER="${2:---both}"

METHODS=(ibp sabr mtl_ibp crown_ibp)
EPS_LABELS=("2_255" "8_255")
EPS_VALUES=("$EPS_2" "$EPS_8")

# Auto-discover the latest ACE model for a given exp-name pattern.
# Searches $ACE_DIR/models_new/cifar10/<exp_name>*/ for the latest net_*.pt
# Usage: find_ace_model_pattern "ibp_2_255" "v3" "sel"
find_ace_model_pattern() {
    local method_eps="$1" version="$2" gate_label="$3"
    # Search for dirs matching: <method>_<eps>[_<gate>]_<version>
    local pattern="${method_eps}"
    if [[ "$gate_label" == "ent" ]]; then
        pattern="${method_eps}_ent"
    fi
    pattern="${pattern}_${version}"

    local base="$ACE_DIR/models_new/cifar10"
    # Find all matching top-level dirs, pick the one with the latest net_*.pt
    local found="" latest_ts=0
    for d in "$base"/${pattern}*/; do
        [[ -d "$d" ]] || continue
        local latest
        latest=$(find "$d" -name "net_*.pt" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1)
        if [[ -n "$latest" ]]; then
            local ts=$(echo "$latest" | awk '{print $1}')
            local path=$(echo "$latest" | awk '{print $2}')
            if (( $(echo "$ts > $latest_ts" | bc -l) )); then
                latest_ts="$ts"
                found="$path"
            fi
        fi
    done

    if [[ -z "$found" ]]; then
        return 1
    fi
    readlink -f "$found"
}

eval_model() {
    local method="$1" eps_label="$2" eps="$3" gate_type="$4"
    local tau grid exp_name model_path

    if [[ "$gate_type" == "net" ]]; then
        grid="$TAU_SEL"
        exp_name="${method}_${eps_label}"
    else
        grid="$TAU_ENT"
        exp_name="${method}_${eps_label}"
    fi

    # Auto-discover the ACE model
    model_path=$(find_ace_model_pattern "$exp_name" "$VERSION" \
        "$( [[ "$gate_type" == "entropy" ]] && echo ent || echo sel )") || {
        echo "  SKIP $method $eps_label $gate_type: no ACE model found"
        return
    }

    # Derive a unique integer exp-id for eval output (ACE requires int)
    local gate_short="sel"
    [[ "$gate_type" == "entropy" ]] && gate_short="ent"
    local eval_exp_name="${method}_${eps_label}_${gate_short}_eval_${VERSION}"
    declare -A _MIDX=( [ibp]=0 [sabr]=1 [mtl_ibp]=2 [crown_ibp]=3 )
    local eidx=0; [[ "$eps_label" == "8_255" ]] && eidx=1
    local gidx=0; [[ "$gate_type" == "entropy" ]] && gidx=1
    local eval_exp_id=$(( 200 + _MIDX[$method]*4 + eidx*2 + gidx ))

    echo "  submitting: eval_${method}_${eps_label}_${gate_type}"
    echo "    model: $model_path"
    echo "    tau: $grid"

    cd "$SCRIPTS_DIR"
    sbatch --job-name="eval_${method}_${eps_label}_${gate_type}_${VERSION}" \
        eval_ace.slurm \
        --load-model "$model_path" \
        --gate-type "$gate_type" \
        --gate-threshold "$grid" \
        --eps "$eps" \
        --exp-id "$eval_exp_id" \
        --exp-name "$eval_exp_name"
}

echo "=== Stage 4: Evaluate ACE compositions ($VERSION) ==="

for method in "${METHODS[@]}"; do
    for i in 0 1; do
        eps_label="${EPS_LABELS[$i]}"
        eps="${EPS_VALUES[$i]}"

        # SelNet
        if [[ "$GATE_FILTER" == "--sel" || "$GATE_FILTER" == "--both" ]]; then
            eval_model "$method" "$eps_label" "$eps" "net"
        fi

        # Entropy
        if [[ "$GATE_FILTER" == "--ent" || "$GATE_FILTER" == "--both" ]]; then
            eval_model "$method" "$eps_label" "$eps" "entropy"
        fi
    done
done

# Smoke test with released branch
echo ""
echo "  --- Smoke test (released C3-IBP branch) ---"
RELEASED_MODEL="$ACE_DIR/trained_models/C3_ACE_Net_IBP_cert_cifar10_2_255.pt"
if [[ -f "$RELEASED_MODEL" ]]; then
    cd "$SCRIPTS_DIR"
    sbatch --job-name="eval_smoke_${VERSION}" \
        eval_ace.slurm \
        --load-model "$RELEASED_MODEL" \
        --gate-type net \
        --gate-threshold "$TAU_SEL" \
        --eps "$EPS_2" \
        --exp-id 299 \
        --exp-name "smoke_${VERSION}"
else
    echo "  SKIP smoke: released model not found ($RELEASED_MODEL)"
fi

echo ""
echo "=== All stage 4 jobs submitted. Check with: squeue -u \$USER ==="

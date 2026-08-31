#!/bin/bash
# ============================================================
# Stage 1: Train all cert networks via CTRAIN
# Usage: bash 01_train_all.sh [version]
#   version = v3 (default), v4, ...
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/00_config.sh"

VERSION="${1:-v1}"

declare -A METHODS=( [ibp]="ibp" [sabr]="sabr" [mtl_ibp]="mtl_ibp" [crown_ibp]="crown_ibp" )

echo "=== Stage 1: Train cert networks ($VERSION) ==="

for method in ibp sabr mtl_ibp crown_ibp; do
    for eps_label in "2_255" "8_255"; do
        if [[ "$eps_label" == "2_255" ]]; then
            EPS="$EPS_2"; EPOCHS="$E2_EPOCHS"; LR="$E2_LR"
        else
            EPS="$EPS_8"; EPOCHS="$E8_EPOCHS"; LR="$E8_LR"
        fi

        JOB_NAME="${method}_${eps_label}_${VERSION}"
        echo "  submitting: $JOB_NAME (method=$method eps=$EPS epochs=$EPOCHS lr=$LR)"

        cd "$SCRIPTS_DIR"
        sbatch --job-name="$JOB_NAME" train_cert_net.slurm \
            --method "$method" \
            --eps "$EPS" \
            --epochs "$EPOCHS" \
            --lr-milestones "$LR"
    done
done

echo ""
echo "=== All stage 1 jobs submitted. Check with: squeue -u \$USER ==="

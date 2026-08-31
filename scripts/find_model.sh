#!/bin/bash
# ============================================================
# Find the latest ACE model checkpoint for a given experiment
# Usage: bash find_model.sh <exp_name> <exp_id>
# Example: bash find_model.sh ibp_2_255_v1 100
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/00_config.sh"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <exp_name> <exp_id>"
    echo ""
    echo "Examples:"
    echo "  $0 ibp_2_255_v1 100        # SelNet selector for IBP 2/255"
    echo "  $0 ibp_2_255_ent_v1 110    # Entropy selector for IBP 2/255"
    echo ""
    echo "Common exp-ids (v1):"
    echo "  Selector training (stage 3):"
    echo "    SelNet: 100-107   (ibp/sabr/mtl/crown × 2/255,8/255)"
    echo "    Entropy: 110-117"
    echo "  Eval (stage 4):"
    echo "    SelNet: 200-207"
    echo "    Entropy: 210-217"
    exit 1
fi

exp_name="$1"
exp_id="$2"

echo "Looking for: $ACE_DIR/models_new/cifar10/$exp_name/$exp_id/"
echo ""

model=$(find_ace_model "$exp_name" "$exp_id") || exit 1

echo "Latest model:"
echo "  $model"

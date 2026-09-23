#!/bin/bash
# ============================================================
# Train the trunk (core-network) with PGD adversarial training.
#
# 8/255 needs the balanced loss (nat-factor 0.5); pure adv training there
# collapses the natural accuracy. 2/255 works with a plain adv loss.
# Recipe matches ACE's core (EfficientNet-B0, CIFAR-10).
#
# (deploy it yourself: copy to ~/research/scripts/.)
#   cd ~/research/ACE && sbatch ~/research/scripts/train_core.slurm \
#       --eps 0.03137254901 --eps-label 8_255 --ver v5
# ============================================================
#SBATCH --job-name=train_core
# account + partition come from config.sh via pipeline.sh. For a direct "sbatch <file>", pass --account/--partition.
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/train_core_%j.out
#SBATCH --error=logs/train_core_%j.err

set -uo pipefail
module load CUDA
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ace
cd "$HOME/research/ACE" || exit 1
mkdir -p "$HOME/research/scripts/logs"

EPS="0.03137254901"; EPS_LABEL="8_255"; VER="v1"; EXP_ID="1"
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --eps)       EPS="$2"; shift 2 ;;
        --eps-label) EPS_LABEL="$2"; shift 2 ;;
        --ver)       VER="$2"; shift 2 ;;
        --exp-id)    EXP_ID="$2"; shift 2 ;;
        *)           EXTRA+=("$1"); shift ;;
    esac
done

# per-eps recipe
if [[ "$EPS_LABEL" == "2_255" ]]; then
    EPOCHS=40; NAT=0.0; STEPS=8
else
    EPOCHS=60; NAT=0.5; STEPS=10
fi

nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
echo "=== train_core $EPS_LABEL ($VER) epochs=$EPOCHS nat=$NAT ==="

python ~/research/scripts/train_core.py \
    --exp-name "core_adv_${EPS_LABEL}_${VER}" \
    --exp-id "$EXP_ID" \
    --train-eps "$EPS" --test-eps "$EPS" \
    --epochs "$EPOCHS" \
    --lr-schedule step --lr-step 5 --lr-factor 0.5 \
    --nat-factor "$NAT" \
    --pgd-steps-train "$STEPS" \
    "${EXTRA[@]}"

echo "=== train_core DONE ==="

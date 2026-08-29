#!/usr/bin/zsh
#SBATCH --job-name=train_selector
#SBATCH --partition=c23g
#SBATCH --account=thes2354
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Train the selection mechanism (gate) for a branch, using ACE's own scripts.
# This is a thin wrapper — no new training logic. ACE does everything:
#   - loads the branch (--load-branch-model)
#   - initializes the gate from it (--gate-mode gate_init)
#   - computes cert targets via diffAI_cert
#   - trains the gate (SelNet: BCE-with-logits; entropy: --cotrain-entropy 0.5)
#
# Usage:
#   # SelectionNet gate
#   sbatch --job-name=sel_sabr2 train_selector.slurm \
#       --branch ../converted/sabr_2_255.pt \
#       --gate-type net --eps 0.00784313725
#
#   # Entropy gate (joint training)
#   sbatch --job-name=sel_sabr2_ent train_selector.slurm \
#       --branch ../converted/sabr_2_255.pt \
#       --gate-type entropy --eps 0.00784313725

module load CUDA
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ace
cd "$HOME/research/ACE" || exit 1

echo "=== train_selector ($(hostname)) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# --- defaults ---
BRANCH=""                   # required: converted branch checkpoint (ACE format)
GATE_TYPE="net"             # net | entropy
EPS="0.00784313725"         # 2/255; use 0.03137254901 for 8/255
BRANCH_NETS="C3_cifar10"
GATE_NETS="C3_cifar10"      # only used for gate-type=net
EXP_ID="1"
COTRAIN_ENTROPY="0.5"       # entropy joint-training weight (ACE default suggestion)

while [[ $# -gt 0 ]]; do
    case $1 in
        --branch) BRANCH="$2"; shift 2 ;;
        --gate-type) GATE_TYPE="$2"; shift 2 ;;
        --eps) EPS="$2"; shift 2 ;;
        --branch-nets) BRANCH_NETS="$2"; shift 2 ;;
        --gate-nets) GATE_NETS="$2"; shift 2 ;;
        --exp-id) EXP_ID="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$BRANCH" ]]; then echo "Error: --branch required"; exit 1; fi

CMD=(python deepTrunk_main.py \
    --train-mode train-diffAI \
    --dataset cifar10 \
    --exp-id "$EXP_ID" \
    --n-branches 1 \
    --gate-type "$GATE_TYPE" \
    --net None \
    --branch-nets "$BRANCH_NETS" \
    --load-branch-model "$BRANCH" \
    --train-batch 100 --test-batch 100 \
    --train-eps "$EPS" --start-eps-factor 1.05 \
    --train-att-n-steps 8 --train-att-step-size 0.25 \
    --test-att-n-steps 40 --test-att-step-size 0.035 \
    --opt adam --lr 1e-3 --lr-step 10 --lr-factor 0.5 \
    --mix --mix-epochs 100 --anneal --anneal-epochs 100 --n-epochs 200 \
    --l1-reg 1e-5 --train-domain box --nat-factor 0.5 \
    --train-trunk False --gate-on-trunk False \
    --retrain-branch False \
    --cert --cert-domain box \
    --test-freq 40 --test-set test)

if [[ "$GATE_TYPE" == "net" ]]; then
    CMD+=(--gate-nets "$GATE_NETS" --gate-mode gate_init --load-gate-model "$BRANCH")
else
    CMD+=(--gate-threshold -0.4 --cotrain-entropy "$COTRAIN_ENTROPY")
fi

"${CMD[@]}"

echo ""
echo "=== train_selector DONE ==="

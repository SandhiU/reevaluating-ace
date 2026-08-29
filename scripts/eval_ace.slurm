#!/usr/bin/zsh
#SBATCH --job-name=eval_ace
#SBATCH --partition=c23g
#SBATCH --account=thes2354
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Evaluate an ACE model (trunk+branch+gate) with ACE's built-in evaluation
# (deepTrunk_main.py --train-mode cert). Sweeps --gate-threshold.
#
# Per-model settings (from ACE GitHub README):
#   --gate-type net     -> SelectionNet gate
#   --gate-type entropy -> Entropy gate (all --gate-nets params ignored)
#   --cert-domain box   -> IBP-trained branches (default relaxation)
#   --cert-domain zono  -> COLT-trained branches
#   --cert-trunk False  -> recommended (trunk not provably trained)
#   --test-set test     -> default; use train to evaluate on train set
#
# τ grids (must match gate type; entropy thresholds are negated):
#   SelectionNet: 0.0,0.3,0.5,0.7,0.9
#   Entropy:      -0.1,-0.3,-0.5,-0.7,-0.9   (⇔ max entropy 0.1 … 0.9)
#   NOTE: entropy τ=-1.0 would mean H ≤ 1.0 nats — outside ACE's released
#   grid ({-0.1 … -0.9}), don't use it for comparability.
#
# Usage — exp-name must match the model family:
#   # IBP branch, SelectionNet gate, eps=2/255
#   sbatch --job-name=eval_ibp2_sel eval_ace.slurm \
#       --load-model ../ace_models/ibp_2_255/ace_model.pt \
#       --gate-type net --gate-threshold 0.0,0.3,0.5,0.7,0.9 \
#       --exp-name ibp_2_255
#
#   # SABR branch, entropy gate, eps=8/255
#   sbatch --job-name=eval_sabr8_ent eval_ace.slurm \
#       --load-model ../ace_models/sabr_8_255/ace_model_entropy.pt \
#       --gate-type entropy --gate-threshold -0.1,-0.3,-0.5,-0.7,-0.9 \
#       --exp-name sabr_8_255_entropy
#
#   # COLT branch, SelectionNet gate -> zono relaxation
#   sbatch --job-name=eval_colt2_sel eval_ace.slurm \
#       --load-model ../ace_models/colt_2_255/ace_model.pt \
#       --gate-type net --cert-domain zono \
#       --gate-threshold 0.0,0.3,0.5,0.7,0.9 \
#       --exp-name colt_2_255

module load CUDA
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ace
cd "$HOME/research/ACE" || exit 1

echo "=== eval_ace ($(hostname)) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# --- defaults ---
LOAD_MODEL=""               # required: branch+gate checkpoint (or full ACE model)
TRUNK_MODEL=""              # trunk checkpoint; auto-derived from --eps if empty
GATE_TYPE="net"             # net | entropy
GATE_THRESHOLD=""           # required, comma-separated
CERT_DOMAIN="box"           # box (IBP) | zono (COLT)
BRANCH_NETS="C3_cifar10"
GATE_NETS="C3_cifar10"      # only used for gate-type=net
NET="efficientnet-b0_pre"
EPS="0.00784313725"         # 2/255; use 0.03137254901 for 8/255
TEST_SET="test"
EXP_ID="1"
EXP_NAME="dev"          # descriptive run label -> models_new/cifar10/<exp_name>/...

while [[ $# -gt 0 ]]; do
    case $1 in
        --load-model) LOAD_MODEL="$2"; shift 2 ;;
        --load-trunk-model) TRUNK_MODEL="$2"; shift 2 ;;
        --gate-type) GATE_TYPE="$2"; shift 2 ;;
        --gate-threshold) GATE_THRESHOLD="$2"; shift 2 ;;
        --cert-domain) CERT_DOMAIN="$2"; shift 2 ;;
        --branch-nets) BRANCH_NETS="$2"; shift 2 ;;
        --gate-nets) GATE_NETS="$2"; shift 2 ;;
        --net) NET="$2"; shift 2 ;;
        --eps) EPS="$2"; shift 2 ;;
        --test-set) TEST_SET="$2"; shift 2 ;;
        --exp-id) EXP_ID="$2"; shift 2 ;;
        --exp-name) EXP_NAME="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$LOAD_MODEL" ]]; then echo "Error: --load-model required"; exit 1; fi
if [[ -z "$GATE_THRESHOLD" ]]; then echo "Error: --gate-threshold required"; exit 1; fi

# Released ACE branch+gate checkpoints contain NO trunk (trained with --net None).
# The trunk must come from --load-trunk-model; auto-derive from eps if not given.
if [[ -z "$TRUNK_MODEL" ]]; then
    if [[ "$EPS" == "0.00784313725" ]]; then
        TRUNK_MODEL="./trained_models/EB-0_cifar10_adv_2_255.pt"
    elif [[ "$EPS" == "0.03137254901" ]]; then
        TRUNK_MODEL="./trained_models/EB-0_cifar10_adv_8_255.pt"
    else
        echo "Error: cannot auto-derive trunk model for eps=$EPS; pass --load-trunk-model"; exit 1
    fi
fi

echo "=== model: $LOAD_MODEL  trunk: $TRUNK_MODEL  gate-type: $GATE_TYPE  cert-domain: $CERT_DOMAIN  eps: $EPS ==="

for TAU in ${(s:,:)GATE_THRESHOLD}; do
    echo ""
    echo "=== gate-type=$GATE_TYPE tau=$TAU cert-domain=$CERT_DOMAIN ==="

    CMD=(python deepTrunk_main.py \
        --train-mode cert \
        --dataset cifar10 \
        --net "$NET" \
        --branch-nets "$BRANCH_NETS" \
        --load-trunk-model "$TRUNK_MODEL" \
        --load-model "$LOAD_MODEL" \
        --n-branches 1 \
        --gate-type "$GATE_TYPE" \
        --gate-threshold "$TAU" \
        --exp-id "$EXP_ID" \
        --exp-name "$EXP_NAME" \
        --train-batch 20 --test-batch 20 \
        --train-eps "$EPS" --test-att-n-steps 40 --test-att-step-size 0.035 \
        --test-set "$TEST_SET" \
        --cert-domain "$CERT_DOMAIN" \
        --cert-trunk False)

    if [[ "$GATE_TYPE" == "net" ]]; then
        CMD+=(--gate-nets "$GATE_NETS")
    fi

    "${CMD[@]}"
done

echo ""
echo "=== eval_ace DONE ==="

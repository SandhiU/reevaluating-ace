#!/usr/bin/zsh
#SBATCH --job-name=train_selector
# account + partition come from config.sh via pipeline.sh. For a direct "sbatch <file>", pass --account/--partition.
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
#       --branch ../converted/sabr_2_255_branch.pt \
#       --gate-type net --eps 0.00784313725
#
#   # Entropy gate (joint training)
#   sbatch --job-name=sel_sabr2_ent train_selector.slurm \
#       --branch ../converted/sabr_2_255_branch.pt \
#       --gate-type entropy --eps 0.00784313725
#
#   # SelectionNet gate trained on alpha-CROWN labels instead of box labels
#   sbatch --job-name=sel_ibp2_alpha train_selector.slurm \
#       --branch ../converted/ibp_2_255_branch.pt --gate-type net --eps 0.00784313725 \
#       --gate-labels-csv ../labels/labels_alpha_crown_ibp_2_255.csv --exp-id 230 --exp-name ibp_2_255_alpha_v3
#
#   # Fast crash test (2 epochs, 50 batches/epoch) — numbers meaningless, exit code matters
#   sbatch --job-name=smoke_ent train_selector.slurm \
#       --branch ../converted/ibp_2_255_branch.pt \
#       --gate-type entropy --eps 0.00784313725 --smoke
#
# ! --branch MUST be a BARE net state dict (keys `blocks.layers.*`), i.e. the
#   *_branch.pt file from convert_format.py. Passing the whole-dTNet checkpoint
#   (keys `branchNet_0.*`) loads ZERO weights silently (load_net_state uses
#   strict=False) -> random branch, nat ~10%, cert 0. Guard below catches it.

module load CUDA
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ace
cd "$HOME/research/ACE" || exit 1

echo "=== train_selector ($(hostname)) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# --- defaults ---
BRANCH=""                   # required: converted branch checkpoint (ACE format)
GATE_MODEL=""                # optional: separate checkpoint for gate init (default: $BRANCH)
GATE_TYPE="net"             # net | entropy
EPS="0.00784313725"         # 2/255; use 0.03137254901 for 8/255
BRANCH_NETS="C3_cifar10"
GATE_NETS="C3_cifar10"      # only used for gate-type=net
EXP_ID="1"
EXP_NAME="dev"          # descriptive label -> models_new/cifar10/<exp_name>/...
COTRAIN_ENTROPY="0.5"       # entropy joint-training weight (ACE default suggestion)
GATE_LABELS_CSV=""          # optional: precomputed gate targets (index,label) -> alpha-CROWN
                            # labels for gate-type=net; needs apply_gate_labels_patch.sh
EPOCHS="200"
MIX_EPOCHS="100"
ANNEAL_EPOCHS="100"
TEST_FREQ="40"
SMOKE="0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --branch) BRANCH="$2"; shift 2 ;;
        --gate-model) GATE_MODEL="$2"; shift 2 ;;
        --gate-type) GATE_TYPE="$2"; shift 2 ;;
        --eps) EPS="$2"; shift 2 ;;
        --branch-nets) BRANCH_NETS="$2"; shift 2 ;;
        --gate-nets) GATE_NETS="$2"; shift 2 ;;
        --exp-id) EXP_ID="$2"; shift 2 ;;
        --exp-name) EXP_NAME="$2"; shift 2 ;;
        --gate-labels-csv) GATE_LABELS_CSV="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --smoke) SMOKE="1"; shift 1 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$BRANCH" ]]; then echo "Error: --branch required"; exit 1; fi

if [[ "$SMOKE" == "1" ]]; then
    EPOCHS="2"; MIX_EPOCHS="1"; ANNEAL_EPOCHS="1"; TEST_FREQ="1"
    echo "=== SMOKE MODE: 2 epochs, fast kappa/eps ramp, --debug (50 batches per pass) ==="
    echo "=== crash test only — the resulting accuracies/certs are MEANINGLESS ==="
fi

# --- guard: reject whole-dTNet checkpoints (they load as 0 keys, silently) ---
python - "$BRANCH" << 'EOF'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu")
bad = [k for k in sd if k.startswith(("branchNet_", "gateNet_", "trunk"))]
if bad:
    sys.exit(f"ERROR: {sys.argv[1]} is a whole-dTNet checkpoint ({len(bad)}/{len(sd)} prefixed "
             "keys). --load-branch-model needs a BARE state dict (blocks.layers.*): use *_branch.pt")
if not any(k.startswith("blocks.") for k in sd):
    sys.exit(f"ERROR: {sys.argv[1]} has no blocks.* keys: {list(sd)[:4]}")
print(f"branch ckpt OK: {len(sd)} bare keys, e.g. {list(sd)[:2]}")
EOF
if [[ $? -ne 0 ]]; then exit 1; fi

CMD=(python deepTrunk_main.py \
    --train-mode train-diffAI \
    --dataset cifar10 \
    --exp-id "$EXP_ID" \
    --exp-name "$EXP_NAME" \
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
    --mix --mix-epochs "$MIX_EPOCHS" --anneal --anneal-epochs "$ANNEAL_EPOCHS" --n-epochs "$EPOCHS" \
    --l1-reg 1e-5 --train-domain box --nat-factor 0.5 \
    --train-trunk False --gate-on-trunk False \
    --retrain-branch False \
    --balanced-gate-loss --sliding-loss-balance 1.0 \
    --cert --cert-domain box \
    --test-freq "$TEST_FREQ" --test-set test)

if [[ "$SMOKE" == "1" ]]; then
    CMD+=(--debug)
fi

if [[ -n "$GATE_LABELS_CSV" ]]; then
    # Precomputed targets replace ACE's internal diffAI_cert y_s for the TRAIN passes.
    # Only meaningful for gate-type=net (see apply_gate_labels_patch.sh).
    echo "=== gate labels injected: $GATE_LABELS_CSV ==="
    CMD+=(--gate-labels-csv "$GATE_LABELS_CSV")
fi

if [[ "$GATE_TYPE" == "net" ]]; then
    CMD+=(--gate-nets "$GATE_NETS" --gate-mode gate_init --load-gate-model "${GATE_MODEL:-$BRANCH}")
else
    CMD+=(--gate-threshold -0.4 --cotrain-entropy "$COTRAIN_ENTROPY")
fi

"${CMD[@]}"
rc=$?

echo ""
if [[ $rc -eq 0 ]]; then
    echo "=== train_selector DONE (exit 0) ==="
else
    echo "=== train_selector FAILED: python exit=$rc — this run did NOT train (see the .err) ==="
fi
# Propagate the python status: without this, the trailing echo makes slurm report
# COMPLETED even when deepTrunk_main.py crashed (bitten 14 Sep: 4 gates 'COMPLETED' in 88 s).
exit $rc

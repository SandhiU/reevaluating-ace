#!/usr/bin/zsh
#SBATCH --job-name=ace_single
#SBATCH --partition=c23g
#SBATCH --account=thes2354
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/ace_%j.out
#SBATCH --error=logs/ace_%j.err

# Run one ACE experiment by ID.
# Usage: sbatch run_experiment.slurm <experiment_id>
# Example: sbatch run_experiment.slurm 36

CONDA_BASE="$HOME/miniforge3"
CONDA_ENV="env_ace"
ACE_CODE_DIR="$HOME/research/ACE/ace_core"
EXPERIMENTS="$HOME/research/scripts/experiments.csv"
TRAINED_DIR="$ACE_CODE_DIR/trained_models"

EXP_ID="${1:-36}"

module load CUDA
source "$CONDA_BASE/bin/activate"
conda activate "$CONDA_ENV"
cd "$ACE_CODE_DIR" || exit 1
mkdir -p logs

LINE=$(grep "^${EXP_ID}," "$EXPERIMENTS" | head -1)
if [[ -z "$LINE" ]]; then
    echo "Error: experiment #$EXP_ID not found"
    exit 1
fi

IFS=',' read -r ID DATASET EPS CORE CKPT BRANCH GATE_NET DOMAIN GTYPE THRESH NAME <<< "$LINE"

echo "=== Experiment #$ID: $NAME ==="
echo "  Threshold: $THRESH | Domain: $DOMAIN | Gate: $GTYPE"
echo "  Host: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

python deepTrunk_main.py \
    --train-mode cert \
    --dataset "$DATASET" \
    --net efficientnet-b0_pre \
    --branch-nets "$BRANCH" \
    --gate-nets "$GATE_NET" \
    --load-trunk-model "$TRAINED_DIR/$CORE" \
    --load-model "$TRAINED_DIR/$CKPT" \
    --n-branches 1 \
    --gate-type "$GTYPE" \
    --gate-threshold "$THRESH" \
    --exp-id "$ID" \
    --exp-name "$NAME" \
    --train-batch 20 --test-batch 20 \
    --train-eps "$EPS" \
    --test-att-n-steps 40 --test-att-step-size 0.035 \
    --test-set test \
    --cert-domain "$DOMAIN" \
    --cert-trunk False

echo "=== Done: $(date) ==="

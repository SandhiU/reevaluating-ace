#!/usr/bin/zsh
#SBATCH --job-name=ace_eval
#SBATCH --partition=c23g
#SBATCH --account=thes2354
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --array=22-25
#SBATCH --output=logs/ace_%A_%a.out
#SBATCH --error=logs/ace_%A_%a.err

# Batch evaluation of ACE models via SLURM array jobs.
# Each array task runs one experiment from experiments.csv
# (task id = data-row number; header and comment lines are skipped).
# Usage: sbatch run_batch.slurm
# Single experiment: sbatch run_experiment.slurm <id>

CONDA_BASE="$HOME/miniforge3"
CONDA_ENV="env_ace"
ACE_DIR="$HOME/research/ACE/ace_core"
EXPERIMENTS="$HOME/research/scripts/experiments_v2.csv"
TRAINED_DIR="$ACE_DIR/trained_models"

module load CUDA
source "$CONDA_BASE/bin/activate"
conda activate "$CONDA_ENV"
cd "$ACE_DIR" || exit 1
mkdir -p logs

# Read experiment #SLURM_ARRAY_TASK_ID from the CSV
LINE=""
COUNT=0
while IFS= read -r L; do
    [[ -z "$L" || "$L" == \#* || "$L" == "id,dataset"* ]] && continue
    ((COUNT++))
    if (( COUNT == SLURM_ARRAY_TASK_ID )); then
        LINE="$L"
        break
    fi
done < "$EXPERIMENTS"

if [[ -z "$LINE" ]]; then
    echo "Error: could not find experiment #$SLURM_ARRAY_TASK_ID"
    echo "Total data rows in CSV: $COUNT"
    exit 1
fi

IFS=',' read -r ID DATASET EPS CORE CKPT BRANCH GATE_NET DOMAIN GTYPE THRESH NAME <<< "$LINE"

echo "========================================"
echo "  Experiment #$ID: $NAME"
echo "========================================"
echo "  Dataset:     $DATASET"
echo "  Epsilon:     $EPS"
echo "  Core:        $CORE"
echo "  Checkpoint:  $CKPT"
echo "  Branch:      $BRANCH"
echo "  Gate net:    $GATE_NET"
echo "  Domain:      $DOMAIN"
echo "  Gate type:   $GTYPE"
echo "  Threshold:   $THRESH"
echo ""
echo "  Host: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not found')"
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
echo "----------------------------------------"

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

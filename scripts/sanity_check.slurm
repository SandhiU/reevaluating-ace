#!/usr/bin/zsh
#SBATCH --job-name=sanity_ibp
#SBATCH --partition=c23g
#SBATCH --account=thes2354
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/sanity_ibp_%j.out
#SBATCH --error=logs/sanity_ibp_%j.err

# CTRAIN sanity check: train IBP on 1000 CIFAR-10 samples for 5 epochs.
# Usage: sbatch sanity_check.slurm

PROJECT_DIR="$HOME/research"
VENV_DIR="$HOME/research/ctrain-env"
SCRIPT="$HOME/research/scripts/sanity_check.py"

module load CUDA

if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: venv not found at $VENV_DIR" >&2
    exit 1
fi

cd "$PROJECT_DIR" || exit 1
mkdir -p logs

echo "=== Job started: $(date) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not found')"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

python "$SCRIPT"

echo "=== Job finished: $(date) ==="

#!/usr/bin/zsh
#SBATCH --job-name=evaluate_ace
#SBATCH --partition=c23g
#SBATCH --account=thes2354
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/evaluate_ace_%j.out
#SBATCH --error=logs/evaluate_ace_%j.err

# Run the upstream evaluate_ACE script (single-model evaluation).
# Usage: sbatch evaluate_ace.slurm

CONDA_BASE="$HOME/miniforge3"
CONDA_ENV="env_ace"
ACE_CODE_DIR="$HOME/research/ACE/ace_core"

module load CUDA
source "$CONDA_BASE/bin/activate"
conda activate "$CONDA_ENV"
cd "$ACE_CODE_DIR" || exit 1
mkdir -p logs

echo "=== Job started: $(date) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not found')"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

bash scripts/evaluate_ACE

echo "=== Job finished: $(date) ==="

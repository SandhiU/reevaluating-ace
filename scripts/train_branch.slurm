#!/usr/bin/zsh
#SBATCH --job-name=ctrain
# account + partition come from config.sh via pipeline.sh. For a direct "sbatch <file>", pass --account/--partition.
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# NOTE: SBATCH lines are read literally by Slurm (no $VAR expansion) — logs are
# relative to the submission dir, so submit from ~/research/scripts after
# `mkdir -p logs`. Give each run a distinct job name on the sbatch line:
#   sbatch --job-name=sabr_2 train_branch.slurm --method sabr --eps 0.00784313725 --epochs 160

module load CUDA
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ace
cd "$HOME/research/scripts" || exit 1

echo "=== train_branch ($(hostname)) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

python train_branch.py "$@"

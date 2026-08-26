#!/usr/bin/zsh
#SBATCH --job-name=ctrain
#SBATCH --partition=c23g
#SBATCH --account=thes2354
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/train_cert_%j.out
#SBATCH --error=logs/train_cert_%j.err
# Submit from ~/research/scripts after mkdir -p logs. Give each run a distinct
# job name on the sbatch line (SBATCH lines are literal — no $VAR expansion).
module load CUDA
source "$HOME/research/ctrain-env/bin/activate"
cd "$HOME/research/scripts" || exit 1
echo "=== train_cert_net ($(hostname)) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}') "
python train_cert_net.py "$@"

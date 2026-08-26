#!/usr/bin/zsh
#SBATCH --job-name=ace_tf32
#SBATCH --partition=c23g
#SBATCH --account=thes2354
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/ace_tf32_%j.out
#SBATCH --error=logs/ace_tf32_%j.err

CONDA_BASE="$HOME/miniforge3"
CONDA_ENV="env_ace"
ACE_CODE_DIR="$HOME/research/ACE/ace_core"
WRAPPER="$HOME/research/scripts/tf32_wrapper.py"

module load CUDA
source "$CONDA_BASE/bin/activate"
conda activate "$CONDA_ENV"
mkdir -p "logs"
cd "$ACE_CODE_DIR" || exit 1

cat > "$WRAPPER" <<'PYEOF'
import os, sys, torch
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
print('TF32 OFF | cudnn:', torch.backends.cudnn.allow_tf32, '| matmul:', torch.backends.cuda.matmul.allow_tf32)
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
import runpy
runpy.run_path(script, run_name='__main__')
PYEOF

echo "=== TF32-off test (exp 45) ==="
echo "  Host: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}') "

python "$WRAPPER" deepTrunk_main.py \
    --train-mode cert \
    --dataset cifar10 \
    --net efficientnet-b0_pre \
    --branch-nets C3_cifar10 \
    --gate-nets C3_cifar10 \
    --load-trunk-model trained_models/EB-0_cifar10_adv_2_255.pt \
    --load-model trained_models/C3_ACE_Net_COLT_cert_cifar10_2_255.pt \
    --n-branches 1 \
    --gate-type net \
    --gate-threshold 0.0 \
    --exp-id 45 \
    --exp-name C3_COLT_tf32off \
    --train-batch 20 --test-batch 20 \
    --train-eps 0.00784313725 \
    --test-att-n-steps 40 --test-att-step-size 0.035 \
    --test-set test \
    --cert-domain zono \
    --cert-trunk False

echo "=== Done: $(date) ==="
echo "Log: $LOG_DIR/ace_tf32_${SLURM_JOB_ID}.out"

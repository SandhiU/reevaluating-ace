#!/bin/bash
# ============================================================
# Standalone core-network evaluation (nat + PGD), bypassing ACE's main.py.
#
# main.py --train-mode test doesn't work for EfficientNet because
# UpscaleNet.determine_dims sets self.dims = None (networks.py:66).
# This script loads the bare EfficientNet-B0 checkpoint directly.
#
# Matches the paper's Appendix F config: EfficientNet-B0, CIFAR-10,
# adversarially trained, PGD 40 steps step 0.035 one restart.
# Paper reports: nat 95.1 / PGD 85.6 at 2/255.
#
#   (deploy it yourself: copy the file to ~/research/scripts/.)
#
# Default (released trunks):
#   cd ~/research/ACE && sbatch ~/research/scripts/eval_core.slurm
#
# Custom checkpoint (pass via --wrap):
#   sbatch --time=0:30:00 --gres=gpu:1 --account <acct> --partition <part> \
#     --job-name=eval_core_v4_8_255 --output=logs/eval_core_v4_8_255_%j.out \
#     --error=logs/eval_core_v4_8_255_%j.err \
#     --wrap="bash ~/research/scripts/eval_core.slurm \
#       'models_new/cifar10/core_adv_8_255/1/efficientnet-b0_pre_0.03137/1789939620/final_model_state.pt' \
#       0.03137254901 core_adv_8_255"
#
# Output: logs/eval_core_<jobid>.out
# ============================================================
#SBATCH --job-name=eval_core
# account + partition come from config.sh via pipeline.sh. For a direct "sbatch <file>", pass --account/--partition.
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/eval_core_%j.out
#SBATCH --error=logs/eval_core_%j.err

set -uo pipefail
module load CUDA
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ace
cd "$HOME/research/ACE" || exit 1
mkdir -p "$HOME/research/scripts/logs"
export PYTHONPATH="$HOME/research/scripts:${PYTHONPATH:-}"

nvidia-smi --query-gpu=name,memory.used --format=csv

# If args are passed: CKPT EPS LABEL [CKPT EPS LABEL ...]
# Otherwise: evaluate the released ACE trunks.
if [[ $# -gt 0 ]]; then
  while [[ $# -ge 3 ]]; do
    CK="$1"; EPS="$2"; LABEL="$3"; shift 3
    echo "===== core $LABEL  eps=$EPS ====="
    if [[ ! -f "$CK" ]]; then
      echo "FAILED: missing $CK" >&2
      continue
    fi
    python "$HOME/research/scripts/eval_core.py" \
      --ckpt "$CK" --eps "$EPS" --label "$LABEL"
  done
else
  for spec in "./trained_models/EB-0_cifar10_adv_2_255.pt 0.00784313725 released_2_255" \
              "./trained_models/EB-0_cifar10_adv_8_255.pt 0.03137254901 released_8_255" \
              "models_new/cifar10/core_adv_2_255/1/efficientnet-b0_pre_0.00784/1789901196/final_model_state.pt 0.00784313725 core_adv_2_255" \
              "models_new/cifar10/core_adv_8_255/3/efficientnet-b0_pre_0.03137/1790067729/best_model.pt 0.03137254901 core_adv_8_255_v3"; do
    set -- $spec
    CK="$1"; EPS="$2"; LABEL="$3"
    echo "===== core $LABEL  eps=$EPS  (paper: nat 95.1 / PGD 85.6 at 2/255) ====="
    if [[ ! -f "$CK" ]]; then
      echo "FAILED: missing $CK" >&2
      ls -l ./trained_models 2>&1 | head -10 >&2
      continue
    fi
    python "$HOME/research/scripts/eval_core.py" \
      --ckpt "$CK" --eps "$EPS" --label "$LABEL"
  done
fi

echo "=== eval_core DONE ==="

#!/bin/bash
# ============================================================
# Standalone gate+branch verification (IBP / CROWN / alpha-CROWN).
# Rebuilds the gate as a plain module so auto_LiRPA's alpha optimizer works.
#
#   bash pipeline.sh gate <target> --in_ver <ver> --ver <ver> --tau 0.0
# ============================================================
#SBATCH --job-name=gen_labels_eval
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/gen_labels_eval_%j.out
#SBATCH --error=logs/gen_labels_eval_%j.err

set -uo pipefail
module load CUDA
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ace
mkdir -p "$HOME/research/scripts/logs"
cd "$HOME/research/ACE" || exit 1
export PYTHONPATH="$HOME/research/ACE:${PYTHONPATH:-}"

nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
echo "=== gen_labels_eval ==="

python "$HOME/research/scripts/gen_labels_eval.py" "$@"

echo "=== gen_labels_eval DONE ==="

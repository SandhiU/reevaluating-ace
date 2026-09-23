#!/bin/bash
# ============================================================
# Gate ROC figure for a trained selector.
# Loads an ACE run dir (args.json + net_*.pt), sweeps tau, writes
# <output>.pdf and <output stem>_bare.pdf (curve only).
#
#   bash pipeline.sh roc <target> --in_ver <ver> --ver <ver> --sel
# ============================================================
#SBATCH --job-name=gen_roc
# account + partition come from config.sh via pipeline.sh. For a direct "sbatch <file>", pass --account/--partition.
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/gen_roc_%j.out
#SBATCH --error=logs/gen_roc_%j.err

set -uo pipefail
module load CUDA
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ace
mkdir -p "$HOME/research/scripts/logs"
# gen_roc imports args_factory/loaders/deepTrunk_networks from ACE, and runs from there
cd "$HOME/research/ACE" || exit 1
export PYTHONPATH="$HOME/research/ACE:${PYTHONPATH:-}"

nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
echo "=== gen_roc ==="

python "$HOME/research/scripts/gen_roc.py" "$@"

echo "=== gen_roc DONE ==="

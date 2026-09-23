#!/bin/bash
# ============================================================
# alpha-CROWN labels for one branch (selector training targets).
# The pipeline submits one job per <method>_<eps>.
#
#   bash pipeline.sh labels <target> --in_ver <ver> --ver <ver> --sel
# ============================================================
#SBATCH --job-name=gen_labels
# account + partition come from config.sh via pipeline.sh. For a direct "sbatch <file>", pass --account/--partition.
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/gen_labels_%j.out
#SBATCH --error=logs/gen_labels_%j.err

set -uo pipefail
module load CUDA
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ace
cd "$HOME/research/ACE" || exit 1
mkdir -p "$HOME/research/scripts/logs" "$HOME/research/labels"

CKPT=""; EPS=""; OUT=""; SAMPLES=50000; BATCH=16; OPT=1; STEPS=100; FORCE=0; EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)       CKPT="$2"; shift 2 ;;
        --eps)        EPS="$2"; shift 2 ;;
        --output)     OUT="$2"; shift 2 ;;
        --samples)    SAMPLES="$2"; shift 2 ;;
        --batch-size) BATCH="$2"; shift 2 ;;
        --opt-steps)  STEPS="$2"; shift 2 ;;
        --no-optimize) OPT=0; shift ;;
        --force)      FORCE=1; shift ;;
        *)            EXTRA+=("$1"); shift ;;
    esac
done
[[ -z "$CKPT" || -z "$OUT" || -z "$EPS" ]] && { echo "need --ckpt --eps --output"; exit 1; }

if [[ -f "$OUT" && "$FORCE" != "1" ]]; then
    echo "SKIP $OUT (already exists; use --force to overwrite)"
    exit 0
fi

OPTFLAG=()
[[ "$OPT" == "1" ]] && OPTFLAG=(--optimize --opt-steps "$STEPS")

nvidia-smi --query-gpu=name,memory.used --format=csv,noheader 2>/dev/null
echo "=== gen_labels  ckpt=$CKPT  eps=$EPS (alpha-CROWN=$OPT) ==="

python "$HOME/research/scripts/gen_labels.py" \
    --ckpt "$CKPT" \
    --eps "$EPS" \
    --output "$OUT" \
    --samples "$SAMPLES" \
    --batch-size "$BATCH" \
    --train \
    "${OPTFLAG[@]}" "${EXTRA[@]}"

echo "=== gen_labels DONE: $OUT ==="

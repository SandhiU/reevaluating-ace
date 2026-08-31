#!/bin/bash
# ============================================================
# Master pipeline runner
# Usage: bash pipeline.sh <stage> [version] [gate_filter]
#
# Stages:
#   train    - Stage 1: train cert networks (CTRAIN)
#   convert  - Stage 2: convert checkpoints to ACE format
#   selector - Stage 3: train selection mechanisms
#   eval     - Stage 4: evaluate ACE compositions
#   all      - Run stages 1→2→3→4 sequentially
#   status   - Show pipeline completion status
#
# Examples:
#   bash pipeline.sh train v1
#   bash pipeline.sh selector v1 --sel
#   bash pipeline.sh eval v1 --ent
#   bash pipeline.sh all v1
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

STAGE="${1:-help}"
VERSION="${2:-v1}"
GATE_FILTER="${3:---both}"

case "$STAGE" in
    train)
        bash "$SCRIPT_DIR/01_train_all.sh" "$VERSION"
        ;;
    convert|bridge)
        bash "$SCRIPT_DIR/02_bridge_all.sh"
        ;;
    selector)
        bash "$SCRIPT_DIR/03_train_selector_all.sh" "$VERSION" "$GATE_FILTER"
        ;;
    eval)
        bash "$SCRIPT_DIR/04_eval_all.sh" "$VERSION" "$GATE_FILTER"
        ;;
    all)
        echo "=========================================="
        echo " Stage 1: Train cert networks"
        echo "=========================================="
        bash "$SCRIPT_DIR/01_train_all.sh" "$VERSION"
        echo ""
        read -p "Wait for training to complete, then press Enter to continue to convert..."
        echo ""

        echo "=========================================="
        echo " Stage 2: Convert checkpoints"
        echo "=========================================="
        bash "$SCRIPT_DIR/02_bridge_all.sh"
        echo ""

        echo "=========================================="
        echo " Stage 3: Train selectors"
        echo "=========================================="
        bash "$SCRIPT_DIR/03_train_selector_all.sh" "$VERSION" "$GATE_FILTER"
        echo ""
        read -p "Wait for selector training to complete, then press Enter to continue to eval..."
        echo ""

        echo "=========================================="
        echo " Stage 4: Evaluate compositions"
        echo "=========================================="
        bash "$SCRIPT_DIR/04_eval_all.sh" "$VERSION" "$GATE_FILTER"
        ;;
    status)
        STATUS_VERSION="${2:-v1}"
        MODELS_DIR="$SCRIPT_DIR/../ACE/models_new/cifar10"

        METHODS=(ibp sabr mtl_ibp crown_ibp)
        EPS=("2_255" "8_255")
        GATES=("sel" "ent")

        echo "Pipeline status for $STATUS_VERSION eval results:"
        echo ""

        if [[ ! -d "$MODELS_DIR" ]]; then
            echo "  No models_new directory found."
        else
            printf "  %-15s" ""
            for eps in "${EPS[@]}"; do
                for gate in "${GATES[@]}"; do
                    printf " %-14s" "${eps}_${gate}"
                done
            done
            echo ""

            for method in "${METHODS[@]}"; do
                printf "  %-15s" "$method"
                for eps in "${EPS[@]}"; do
                    for gate in "${GATES[@]}"; do
                            # Find eval dirs: match _sel_ or _ent_ as delimited tokens
                        found=0
                        for d in "$MODELS_DIR"/*${method}_${eps}_${gate}_${STATUS_VERSION}*/ "$MODELS_DIR"/*${method}_${eps}_${gate}_eval_${STATUS_VERSION}*/; do
                            [[ -d "$d" ]] || continue
                            count=$(find "$d" -name "cert_log.csv" 2>/dev/null | wc -l)
                            if [[ "$count" -gt 0 ]]; then
                                printf " %-14s" "[done] ($count)"
                                found=1
                                break
                            fi
                        done
                        if [[ "$found" == "0" ]]; then
                            printf " %-14s" "[---]"
                        fi
                    done
                done
                echo ""
            done
            echo ""
            echo "  [done] = eval complete  [---] = not started"
        fi
        ;;
    help|*)
        echo "Usage: $0 <stage> [version] [gate_filter]"
        echo ""
        echo "Stages:"
        echo "  train     Train cert networks (CTRAIN)"
        echo "  convert   Convert checkpoints to ACE format"
        echo "  selector  Train selection mechanisms"
        echo "  eval      Evaluate ACE compositions"
        echo "  all       Run stages 1→2→3→4"
        echo "  status    Show pipeline completion"
        echo ""
        echo "Gate filters: --sel, --ent, --both (default)"
        echo "Versions: v1 (default), v2, v3, ..."
        echo ""
        echo "Examples:"
        echo "  $0 train v1"
        echo "  $0 selector v1 --sel"
        echo "  $0 eval v1 --ent"
        echo "  $0 status v1"
        ;;
esac

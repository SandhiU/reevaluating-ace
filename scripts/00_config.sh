#!/bin/bash
# ============================================================
# ACE pipeline — shared config
# All paths are HPC-relative (~/research/...).
# ============================================================

RESEARCH_DIR="$HOME/research"
RUNS_DIR="$RESEARCH_DIR/runs"
CONVERTED_DIR="$RESEARCH_DIR/converted"
SCRIPTS_DIR="$RESEARCH_DIR/scripts"
ACE_DIR="$RESEARCH_DIR/ACE"

# Training hyperparameters (SABR paper Table 5)
E2_EPOCHS=160; E2_LR="120,140"   # 2/255
E8_EPOCHS=180; E8_LR="140,160"   # 8/255

# Epsilon values
EPS_2="0.00784313725"   # 2/255
EPS_8="0.03137254901"   # 8/255

# Tau grids
# Certifiable selection needs lb(gate) >= tau, so NEGATIVE tau is the easier direction
# (runbook, 31 Aug). Positive-only grid is strictly harder than tau=0 and only reduces routing.
TAU_SEL="-0.9,-0.5,-0.2,0.0,0.3,0.5"
TAU_ENT="-0.1,-0.3,-0.5,-0.7,-0.9"

# ACE architecture
BRANCH_NETS="C3_cifar10"
GATE_NETS="C3_cifar10"
NET="efficientnet-b0_pre"

# Find the latest ACE model checkpoint for a given exp-name and exp-id
# Usage: find_ace_model <exp_name> <exp_id>
find_ace_model() {
    local exp_name="$1" exp_id="$2"
    local base="$ACE_DIR/models_new/cifar10/$exp_name/$exp_id"
    if [[ ! -d "$base" ]]; then
        echo "ERROR: directory not found: $base" >&2; return 1
    fi
    # Find the latest net_*.pt across all subdirs
    local model
    model=$(find "$base" -name "net_*.pt" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}')
    if [[ -z "$model" ]]; then
        echo "ERROR: no net_*.pt found under $base" >&2; return 1
    fi
    readlink -f "$model"
}

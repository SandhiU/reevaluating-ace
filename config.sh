#!/usr/bin/env bash
# ============================================================
# config.sh -- shared config for the pipeline
# Sourced by pipeline.sh and pipeline_status.sh. All paths are HPC-relative.
# See research/README.md for the pipeline overview and thesis Ch.3 for the method.
# ============================================================

RESEARCH_DIR="$HOME/research"
RUNS_DIR="$RESEARCH_DIR/runs"                # CTRAIN branch outputs
CONVERTED_DIR="$RESEARCH_DIR/converted"      # branches in ACE format
LABELS_DIR="$RESEARCH_DIR/labels"            # alpha-CROWN selector targets
FIG_DIR="$RESEARCH_DIR/figures"
RESULTS_DIR="$RESEARCH_DIR/results"          # aggregated result CSVs
SCRIPTS_DIR="$RESEARCH_DIR/scripts"
ACE_DIR="$RESEARCH_DIR/ACE"

DEFAULT_VER="v5"

# ---- SLURM (site-specific; override via env or edit here) ----
# #SBATCH lines cannot read shell vars, so the pipeline passes these on the sbatch
# command line. Set SLURM_ACCOUNT/SLURM_PARTITION for your cluster; leave empty to
# use the cluster default.
SLURM_ACCOUNT="${SLURM_ACCOUNT:-thes2354}"
SLURM_PARTITION="${SLURM_PARTITION:-c23g}"

# ---- epsilon -------------------------------------------------------------
EPS_2="0.00784313725"    # 2/255
EPS_8="0.03137254901"    # 8/255

# ---- tau grids -----------------------------------------------------------
# certifiable routing needs lb(gate) >= tau, so negative tau is the easy side.
# 2.0 and -2.0 bracket the front; 0.2..1.5 fill in the high-natural-accuracy end.
TAU_SEL="-2.0,-1.5,-1.0,-0.7,-0.5,-0.2,0.0,0.2,0.5,0.8,1.0,1.2,1.5,2.0"
TAU_ENT="-0.9,-0.7,-0.5,-0.3,-0.1"

# ---- trunks --------------------------------------------------------------
TRUNK_MODEL_2="$ACE_DIR/trained_models/EB-0_cifar10_adv_2_255.pt"
TRUNK_MODEL_8="$ACE_DIR/trained_models/EB-0_cifar10_adv_8_255.pt"
# our retrained cores (used from v4 onwards)
TRUNK_MODEL_2_NEW="$ACE_DIR/models_new/cifar10/core_adv_2_255/1/efficientnet-b0_pre_0.00784/1789901196/final_model_state.pt"
TRUNK_MODEL_8_NEW=$(find "$ACE_DIR/models_new/cifar10/core_adv_8_255/3" -name 'best_model.pt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}' || true)
if [[ -z "$TRUNK_MODEL_8_NEW" ]]; then
    TRUNK_MODEL_8_NEW=$(find "$ACE_DIR/models_new/cifar10/core_adv_8_255" -name 'final_model_state.pt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}' || true)
fi

# ---- branch training hyperparameters -------------------------------------
# fill in from the supervisor's values; branch_hp returns the extra flags for
# train_branch.slurm. Keep --epochs and --lr-milestones, add --alpha where the
# method needs it (SABR lambda, MTL-IBP alpha).
# ---- branch training hyperparameters -------------------------------------
# Best values per method and epsilon (from the supervisor). branch_hp returns the
# flags for train_branch.slurm: --epochs, --lr-milestones, --alpha, and the CTRAIN
# wrapper kwargs via --set. Keys must match the CTRAIN *ModelWrapper signature.
branch_hp() {  # branch_hp <method> <eps_label>
    case "$1_$2" in
        ibp_2_255)
            echo "--epochs 160 --lr-milestones 120,140 --set warm_up_epochs=1 --set ramp_up_epochs=80 --set lr_decay_factor=0.2 --set l1_reg_weight=1e-8 --set shi_reg_weight=0.5 --set train_eps_factor=1" ;;
        ibp_8_255)
            echo "--epochs 160 --lr-milestones 120,140 --set warm_up_epochs=1 --set ramp_up_epochs=80 --set lr_decay_factor=0.2 --set l1_reg_weight=0 --set shi_reg_weight=0.5 --set train_eps_factor=1" ;;
        sabr_2_255)
            echo "--epochs 160 --lr-milestones 120,140 --alpha 0.1 --set warm_up_epochs=1 --set ramp_up_epochs=80 --set l1_reg_weight=1e-6 --set shi_reg_weight=0.5 --set pgd_steps=8 --set pgd_restarts=1 --set pgd_alpha=0.25 --set pgd_decay_milestones=none --set pgd_eps_factor=2.1" ;;
        sabr_8_255)
            echo "--epochs 180 --lr-milestones 140,160 --alpha 0.7 --set warm_up_epochs=1 --set ramp_up_epochs=80 --set l1_reg_weight=0 --set shi_reg_weight=0.5 --set pgd_steps=8 --set pgd_restarts=1 --set pgd_alpha=0.5 --set pgd_decay_milestones=4,7 --set pgd_alpha_decay_factor=0.1 --set pgd_eps_factor=1" ;;
        mtl_ibp_2_255)
            echo "--epochs 160 --lr-milestones 120,140 --alpha 0.004 --set warm_up_epochs=1 --set ramp_up_epochs=80 --set lr_decay_factor=0.2 --set l1_reg_weight=3e-6 --set shi_reg_weight=0.5 --set train_eps_factor=1 --set pgd_steps=8 --set pgd_alpha=0.25 --set pgd_eps_factor=2.1" ;;
        mtl_ibp_8_255)
            echo "--epochs 260 --lr-milestones 180,220 --alpha 0.5 --set warm_up_epochs=1 --set ramp_up_epochs=80 --set lr_decay_factor=0.2 --set l1_reg_weight=1e-7 --set shi_reg_weight=0.5 --set train_eps_factor=1 --set pgd_steps=1 --set pgd_alpha=10 --set pgd_eps_factor=1" ;;
        crown_ibp_2_255)
            echo "--epochs 160 --lr-milestones 120,140 --set warm_up_epochs=1 --set ramp_up_epochs=80 --set lr_decay_factor=0.2 --set l1_reg_weight=0 --set shi_reg_weight=0.5 --set train_eps_factor=1 --set start_beta=1 --set end_beta=1" ;;
        crown_ibp_8_255)
            echo "--epochs 160 --lr-milestones 120,140 --set warm_up_epochs=1 --set ramp_up_epochs=80 --set lr_decay_factor=0.2 --set l1_reg_weight=0 --set shi_reg_weight=0.5 --set train_eps_factor=1 --set start_beta=1 --set end_beta=1" ;;
        *) echo "--epochs 160 --lr-milestones 120,140" ;;
    esac
}

# ---- exp-id scheme -------------------------------------------------------
# stable per method/eps/gate; the version lives in the exp-name, not the id.
# method: ibp=0 sabr=1 mtl_ibp=2 crown_ibp=3 ; eps: 2_255=0 8_255=1 ; gate: sel=0 ent=1
_method_idx() { case "$1" in ibp) echo 0 ;; sabr) echo 1 ;; mtl_ibp) echo 2 ;; crown_ibp) echo 3 ;; *) echo 0 ;; esac; }
_eps_idx()    { [[ "$1" == 8_255 ]] && echo 1 || echo 0; }
_gate_idx()   { [[ "$1" == ent ]] && echo 1 || echo 0; }
sel_id()   { echo $(( 300 + $(_method_idx "$1")*10 + $(_eps_idx "$2")*2 + $(_gate_idx "$3") )); }
eval_id()  { echo $(( 400 + $(_method_idx "$1")*10 + $(_eps_idx "$2")*2 + $(_gate_idx "$3") )); }

# ---- paths ---------------------------------------------------------------
run_dir()    { echo "$RUNS_DIR/$1_$2_$3"; }                     # <method>_<eps>_<ver>
branch_pt()  { echo "$CONVERTED_DIR/$1_$2_$3_branch.pt"; }
label_csv()  { echo "$LABELS_DIR/labels_alpha_crown_$1_$2_$3.csv"; }

# resolve an existing input branch/labels: versioned name first, then the
# unversioned name the pre-v5 pipeline used (converted/<m>_<e>_branch.pt etc).
branch_in() {
    local p; p="$(branch_pt "$1" "$2" "$3")"
    [[ -f "$p" ]] && { echo "$p"; return 0; }
    local u="$CONVERTED_DIR/$1_$2_branch.pt"
    [[ -f "$u" ]] && echo "$u"
    return 0
}
label_in() {
    local p; p="$(label_csv "$1" "$2" "$3")"
    [[ -f "$p" ]] && { echo "$p"; return 0; }
    local u="$LABELS_DIR/labels_alpha_crown_$1_$2.csv"
    [[ -f "$u" ]] && echo "$u"
    return 0
}
sel_name() {  # <method> <eps> <ver> <gate>; entropy carries an _ent tag
    if [[ "$4" == ent ]]; then echo "$1_$2_ent_$3"; else echo "$1_$2_$3"; fi
}
eval_name() {
    local gs=sel; [[ "$3" == ent ]] && gs=ent
    if [[ -n "${GATE_MODEL:-}" && "$GATE_MODEL" != "$1" ]]; then
        echo "$1_$2_xgate-${GATE_MODEL}_${gs}_eval_$4"
    else
        echo "$1_$2_${gs}_eval_$4"
    fi
}

# ---- discovery -----------------------------------------------------------
# find_selector <exp_name> -> latest ACE run dir with net_*.pt, or empty
find_selector() {
    local exp_name="$1" base="$ACE_DIR/models_new/cifar10"
    local f
    f=$(find "$base"/${exp_name}/ -name 'net_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1)
    [[ -z "$f" ]] && return 0
    dirname "$(echo "$f" | awk '{print $2}')"
}
latest_net() { find "$1" -name 'net_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}'; }

# ---- misc ----------------------------------------------------------------
trunk_for() { [[ "$1" == 2_255 ]] && echo "$TRUNK_MODEL_2_NEW" || echo "$TRUNK_MODEL_8_NEW"; }
tau_grid()  { [[ "$1" == ent ]] && echo "$TAU_ENT" || echo "$TAU_SEL"; }

# verify_domains <box|alpha|alpha-gate|all> -> --cert-domain values for eval_ace.slurm
# ACE passes the SAME domain to the gate and the branch, so alpha-CROWN on the gate
# and on the branch are the same run; alpha and alpha-gate both map to l_alpha.
verify_domains() {
    case "$1" in
        box)        echo "box" ;;
        alpha)      echo "l_alpha" ;;
        alpha-gate) echo "l_alpha" ;;
        all)        echo "box l_alpha" ;;
        *) echo "unknown --verify: $1" >&2; exit 1 ;;
    esac
}

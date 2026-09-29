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

# ---- verification (labels_eval stage) ------------------------------------
# The alpha-CROWN backward scales with batch size and accumulates: ACE's default
# test_batch=100 OOMs (~80 GiB) inside a single batch. gen_labels.py measured 16 as
# the safe H100 value, and gen_labels_eval.py now batches + frees per batch. Override
# per run either via env (LABELS_EVAL_BATCH=8 bash pipeline.sh ...) or --batch-size.
LABELS_EVAL_BATCH="${LABELS_EVAL_BATCH:-16}"
LABELS_EVAL_MEM_LIMIT="${LABELS_EVAL_MEM_LIMIT:-60}"   # clean exit 2 above this many GiB

# ---- trunks --------------------------------------------------------------
TRUNK_MODEL_2="$ACE_DIR/trained_models/EB-0_cifar10_adv_2_255.pt"
TRUNK_MODEL_8="$ACE_DIR/trained_models/EB-0_cifar10_adv_8_255.pt"
# Our retrained cores. Resolve the newest run automatically, so a retrain
#   bash pipeline.sh train_core --eps 8_255 --ver <ver>
# is picked up by convert/labels/eval without editing paths here. best_model.pt wins
# over final_model_state.pt; newest timestamp wins within each.
_latest_core() {  # _latest_core <core_adv_eps...>
    local root="$ACE_DIR/models_new/cifar10" f name
    for name in best_model.pt final_model_state.pt; do
        f=$( { find "$root" -path "*/$1*/$name" -printf '%T@ %p\n' 2>/dev/null || true; } \
             | sort -rn | head -1 | awk '{print $2}' )
        if [[ -n "$f" ]]; then echo "$f"; return 0; fi
    done
    return 0
}
TRUNK_MODEL_2_NEW="$(_latest_core core_adv_2_255)"
TRUNK_MODEL_8_NEW="$(_latest_core core_adv_8_255)"

# ---- released ACE compositions (the control side) ------------------------
# `released` / `roc_released` stages. Each file below is a released ACE model
# (branch + SelectionNet gate, trained with --net None -> it carries NO trunk;
# we supply the trunk with --load-trunk-model).
#   IBP  -> box  cert domain   (released IBP branches)
#   COLT -> zono cert domain   (ACE certifies COLT-trained branches with zonotopes)
REL_MODEL() {  # REL_MODEL <method> <eps_label>
    case "$1_$2" in
        ibp_2_255)  echo "$ACE_DIR/trained_models/C3_ACE_Net_IBP_cert_cifar10_2_255.pt" ;;
        ibp_8_255)  echo "$ACE_DIR/trained_models/C3_ACE_Net_IBP_cert_cifar10_8_255.pt" ;;
        colt_2_255) echo "$ACE_DIR/trained_models/C3_ACE_Net_COLT_cert_cifar10_2_255.pt" ;;
        colt_8_255) echo "$ACE_DIR/trained_models/C3_ACE_Net_COLT_cert_cifar10_8_255.pt" ;;
        *) echo "" ;;
    esac
}
rel_cert_domain() { [[ "$1" == colt ]] && echo zono || echo box; }
rel_methods() {     # `all` covers both control methods; --eps picks the epsilon
    case "$1" in
        all)        echo "ibp colt" ;;
        ibp|colt)   echo "$1" ;;
        *) echo "unknown released target: $1 (use ibp|colt|all)" >&2; exit 1 ;;
    esac
}
rel_id() {  # stable exp-ids, 500+ (no clash with selector 300+/eval 400+)
    case "$1_$2" in ibp_2_255) echo 500 ;; ibp_8_255) echo 501 ;; colt_2_255) echo 502 ;; colt_8_255) echo 503 ;; *) echo 500 ;; esac
}

# ---- branch training hyperparameters -------------------------------------
# fill in from the supervisor's values; branch_hp returns the extra flags for
# train_branch.slurm. Keep --epochs and --lr-milestones, add --alpha where the
# method needs it (SABR lambda, MTL-IBP alpha).
# ---- core (trunk) training recipe ----------------------------------------
# core_hp returns the extra flags for train_core.slurm. The Madry PGD step size is
# eps/4 and train_core.py now defaults to it, so it is not set here. --nat-factor
# blends clean/adv loss (0 = pure adv, 0.5 = balanced). Too large a nat-factor, or a
# PGD step well above eps/4, collapses robustness; keep the step near eps/4.
core_hp() {  # core_hp <eps_label>
    case "$1" in
        2_255) echo "--nat-factor 0.0 --pgd-steps-train 8 --epochs 40" ;;
        8_255) echo "--nat-factor 0.2 --pgd-steps-train 20 --epochs 80" ;;
        *)     echo "" ;;
    esac
}

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
# unversioned fallback name (converted/<m>_<e>_branch.pt etc).
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

# CTRAIN raw branch (runs/<method>_<eps>_<ver>/final_model_state.pt). gen_labels.py needs
# the plain nn.Sequential, not the ACE-format branch (keys 0.weight vs blocks.layers.*).
ctrain_in() {
    local eps; eps=$( [[ "$2" == 2_255 ]] && echo "$EPS_2" || echo "$EPS_8" )
    local v; v=$(printf '%s/final_model_state.pt' "$(run_dir "$1" "$2" "$3")")
    [[ -f "$v" ]] && { echo "$v"; return 0; }
    local u; u=$(printf "$RUNS_DIR/${1}_eps%.8f/final_model_state.pt" "$eps")
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

# verify_domains <box|alpha|all> -> --cert-domain values for eval_ace.slurm
# alpha runs alpha-CROWN on the gate AND the branch (ACE passes one cert domain to both,
# see ai_cert_sample), so there is no separate gate-only mode.
verify_domains() {
    case "$1" in
        box)   echo "box" ;;
        alpha) echo "l_alpha" ;;
        all)   echo "box l_alpha" ;;
        *) echo "unknown --verify: $1" >&2; exit 1 ;;
    esac
}

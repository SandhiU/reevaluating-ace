#!/usr/bin/env bash
# ============================================================
# pipeline.sh -- ACE pipeline runner
#
# Usage:  bash pipeline.sh <stage> <target> [flags]
#
# Stages:
#   train_core   train the trunk (core-network) with PGD adv training
#   eval_core    evaluate a retrained core (nat + PGD at --eps)
#   train        train the branch with CTRAIN
#   convert      CTRAIN checkpoint -> ACE format
#   labels       alpha-CROWN labels for the branch (selector targets)
#   selector     train the selection mechanism on those labels
#   roc          gate ROC figure from the trained selector
#   roc_released gate ROC figure for a RELEASED ACE model (ibp|colt)
#   labels_eval  gate + branch verification (IBP / CROWN / alpha-CROWN)
#   cross_gate   cross-gate eval: gate from one method, branch from another
#   eval         evaluate the composition (box / alpha-CROWN)
#   released     evaluate a RELEASED ACE composition (ibp|colt) on our trunk
#   aggregate    collect cert_log.csv into results/agg_<ver>.csv
#   all          run the branch pipeline end to end (train -> ... -> aggregate)
#   status       show what each stage has produced
#
# Targets:  ibp | sabr | mtl-ibp | crown-ibp | all
#           (train_core ignores the method target and uses --eps)
#
# Every stage reads its input from --in_ver and writes under --ver, so stages
# can be run on their own against any earlier version. A full run with one
# version is just --ver X (in_ver defaults to X).
#
# Flags:
#   --ver VER            output version tag (default $DEFAULT_VER)
#   --out_ver VER        alias for --ver
#   --in_ver VER         input version tag (default: same as --ver)
#   --gate-model METHOD  cross-gate eval: gate trained for another method
#   --in-gate-model VER  version tag of the gate model (default: --in_ver)
#   --sel | --ent        selector type (default --sel)
#   --eps SET            2_255 | 8_255 | both (default both)
#   --verify MODE        box | alpha | all (default box)
#   --batch-size N       labels_eval verifier batch size (default 16; small on purpose,
#                        the alpha-CROWN backward OOMs at ACE's default 100)
#   --tau VALUE          override the gate-threshold grid (e.g. --tau 0.0)
#   --kind MODE          aggregate: all | eval | train (default all)
#   --trunk-model PATH   explicit trunk override
#   --wait               block between stages until squeue drains
#   --dry-run            print the sbatch lines, do not submit
#
# Examples:
#   bash pipeline.sh train_core --eps 8_255 --ver <ver>
#   bash pipeline.sh train sabr --ver <ver> --sel
#   bash pipeline.sh labels all --in_ver <ver> --ver <ver>
#   bash pipeline.sh labels_eval all --in_ver <ver> --ver <ver> --eps 2_255
#   bash pipeline.sh eval mtl-ibp --gate-model ibp --in_ver <ver> --in-gate-model <ver> --ver <ver> --sel
#   bash pipeline.sh eval all --in_ver <ver> --ver <ver>
#   bash pipeline.sh released colt --eps 8_255 --ver <ver>
#   bash pipeline.sh roc_released all --eps 2_255 --ver <ver>
#   bash pipeline.sh all ibp --ver <ver> --in_ver <ver> --sel --verify all
#   bash pipeline.sh aggregate --in_ver <ver>
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

STAGE="${1:-help}"; shift || true
if [[ $# -gt 0 && "${1}" != -* ]]; then TARGET="$1"; shift; else TARGET="all"; fi

VER="$DEFAULT_VER"
IN_VER=""
GATE_MODEL=""
IN_GATE_MODEL=""
GATE="sel"
EPS_SET="both"
VERIFY="box"
KIND="all"
TAU_OVERRIDE=""
TRUNK_OVERRIDE=""
DRY=0
WAIT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ver|--out_ver)  VER="$2"; shift 2 ;;
        --in_ver)         IN_VER="$2"; shift 2 ;;
        --gate-model)     GATE_MODEL="$2"; shift 2 ;;
        --in-gate-model)  IN_GATE_MODEL="$2"; shift 2 ;;
        --sel)            GATE="sel"; shift ;;
        --ent)            GATE="ent"; shift ;;
        --eps)            EPS_SET="$2"; shift 2 ;;
        --verify)         VERIFY="$2"; shift 2 ;;
        --tau)            TAU_OVERRIDE="$2"; shift 2 ;;
        --batch-size)     LABELS_EVAL_BATCH="$2"; shift 2 ;;
        --kind)           KIND="$2"; shift 2 ;;
        --trunk-model)    TRUNK_OVERRIDE="$2"; shift 2 ;;
        --wait)           WAIT=1; shift ;;
        --dry-run)        DRY=1; shift ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done
[[ -z "$IN_VER" ]] && IN_VER="$VER"
[[ -z "$IN_GATE_MODEL" ]] && IN_GATE_MODEL="$IN_VER"

run() {  # run <job-name> <script> <args...>
    local job="$1" script="$2"; shift 2
    local sb=(--account "$SLURM_ACCOUNT" --partition "$SLURM_PARTITION")
    [[ -z "$SLURM_ACCOUNT" ]] && sb=(--partition "$SLURM_PARTITION")
    if [[ "$DRY" == "1" ]]; then
        echo "  [dry] sbatch --job-name=$job ${sb[*]} $script $*"
    else
        ( cd "$SCRIPT_DIR" && sbatch --job-name="$job" "${sb[@]}" "$script" "$@" )
    fi
}
py() {  # py <args...>  (local, prints on dry-run)
    if [[ "$DRY" == "1" ]]; then echo "  [dry] python $*"; else python "$@"; fi
}
wait_jobs() {
    [[ "$WAIT" == "1" && "$DRY" != "1" ]] || return 0
    echo "  waiting for jobs to drain ..."
    while squeue -u "$USER" -h 2>/dev/null | grep -q .; do sleep 60; done
}

methods() {
    case "$1" in
        all)                        echo "ibp sabr mtl_ibp crown_ibp" ;;
        mtl-ibp|mtl_ibp)            echo "mtl_ibp" ;;
        ibp|sabr|crown-ibp|crown_ibp) echo "${1/-/_}" ;;
        *) echo "unknown target: $1" >&2; exit 1 ;;
    esac
}
eps_set() {
    case "$1" in
        both)        echo "2_255 8_255" ;;
        2_255|8_255) echo "$1" ;;
        *) echo "unknown --eps: $1" >&2; exit 1 ;;
    esac
}
eps_of() { [[ "$1" == 2_255 ]] && echo "$EPS_2" || echo "$EPS_8"; }

# ---- stages ------------------------------------------------------------

stage_train_core() {
    echo "== train_core ($VER) eps=$EPS_SET =="
    for e in $(eps_set "$EPS_SET"); do
        run "core_${e}_${VER}" train_core.slurm \
            --eps "$(eps_of "$e")" --eps-label "$e" --ver "$VER" --exp-id 1 \
            $(core_hp "$e")
    done
}

stage_eval_core() {
    echo "== eval_core ($VER) eps=$EPS_SET =="
    for e in $(eps_set "$EPS_SET"); do
        local ck; ck="$(_latest_core "core_adv_${e}")"
        [[ -z "$ck" ]] && { echo "  skip eval_core $e: no core run (core_adv_${e}*)"; continue; }
        echo "  $e -> $ck"
        run "eval_core_${e}_${VER}" eval_core.slurm \
            "$ck" "$(eps_of "$e")" "core_adv_${e}_${VER}"
    done
}

stage_train() {
    echo "== train ($VER) target=$TARGET eps=$EPS_SET =="
    for m in $(methods "$TARGET"); do
        for e in $(eps_set "$EPS_SET"); do
            run "${m}_${e}_${VER}" train_branch.slurm \
                --method "$m" --eps "$(eps_of "$e")" \
                --run-dir "$(run_dir "$m" "$e" "$VER")" \
                $(branch_hp "$m" "$e")
        done
    done
    echo "  training submitted; run 'bash pipeline.sh convert $TARGET --in_ver $VER --ver $VER' once it finishes"
}

stage_convert() {
    echo "== convert ($VER) target=$TARGET =="
    for m in $(methods "$TARGET"); do
        for e in $(eps_set "$EPS_SET"); do
            local src; src="$(run_dir "$m" "$e" "$IN_VER")"
            [[ -d "$src" ]] || { echo "  skip convert $m $e: no ${src}"; continue; }
            py "$SCRIPT_DIR/convert_format.py" --method "$m" --eps "$(eps_of "$e")" \
                --runs-dir "$src" --output "$(branch_pt "$m" "$e" "$VER")"
        done
    done
}

stage_labels() {
    echo "== labels ($VER, in=$IN_VER) target=$TARGET =="
    for m in $(methods "$TARGET"); do
        for e in $(eps_set "$EPS_SET"); do
            local ck; ck="$(ctrain_in "$m" "$e" "$IN_VER")"
            [[ -z "$ck" ]] && { echo "  skip labels $m $e: no CTRAIN model for in=$IN_VER"; continue; }
            run "labels_${m}_${e}_${VER}" gen_labels.slurm \
                --ckpt "$ck" \
                --eps "$(eps_of "$e")" \
                --output "$(label_csv "$m" "$e" "$VER")"
        done
    done
}

stage_selector() {
    echo "== selector ($GATE) ($VER, in=$IN_VER) target=$TARGET =="
    for m in $(methods "$TARGET"); do
        for e in $(eps_set "$EPS_SET"); do
            local br lb; br="$(branch_in "$m" "$e" "$IN_VER")"; lb="$(label_in "$m" "$e" "$IN_VER")"
            [[ -z "$br" ]] && { echo "  skip selector $m $e: no branch for in=$IN_VER"; continue; }
            [[ -z "$lb" ]] && { echo "  skip selector $m $e: no labels for in=$IN_VER"; continue; }
            run "${GATE}_${m}_${e}_${VER}" train_selector.slurm \
                --branch "$br" \
                --gate-type "$( [[ "$GATE" == ent ]] && echo entropy || echo net )" \
                --gate-labels-csv "$lb" \
                --eps "$(eps_of "$e")" \
                --exp-id "$(sel_id "$m" "$e" "$GATE")" \
                --exp-name "$(sel_name "$m" "$e" "$VER" "$GATE")"
        done
    done
}

stage_roc() {
    echo "== roc ($VER, in=$IN_VER) target=$TARGET =="
    for m in $(methods "$TARGET"); do
        for e in $(eps_set "$EPS_SET"); do
            local dir; dir="$(find_selector "$(sel_name "$m" "$e" "$IN_VER" "$GATE")")"
            [[ -z "$dir" ]] && { echo "  skip roc $m $e: no selector $(sel_name "$m" "$e" "$IN_VER" "$GATE")"; continue; }
            run "roc_${m}_${e}_${VER}" gen_roc.slurm \
                --run-dir "$dir" --eps "$(eps_of "$e")" \
                --output "$FIG_DIR/roc_${m}_${e}_${VER}.pdf"
        done
    done
}

stage_labels_eval() {
    echo "== labels_eval ($VER, in=$IN_VER) target=$TARGET =="
    for m in $(methods "$TARGET"); do
        for e in $(eps_set "$EPS_SET"); do
            local name; name="$(sel_name "$m" "$e" "$IN_VER" "$GATE")"
            local dir;  dir="$(find_selector "$name")"
            [[ -z "$dir" ]] && { echo "  skip labels_eval $m $e: no selector $name"; continue; }
            mkdir -p "$RESULTS_DIR/dumps"
            run "labels_eval_${m}_${e}_${VER}" gen_labels_eval.slurm \
                --run-dir "$dir" --eps "$(eps_of "$e")" \
                --samples 10000 \
                --batch-size "$LABELS_EVAL_BATCH" \
                --mem-limit-gib "$LABELS_EVAL_MEM_LIMIT" \
                --dump-npz "$RESULTS_DIR/dumps/labels_eval_${m}_${e}_${IN_VER}.npz"
        done
    done
}

stage_eval() {
    echo "== eval ($VER, in=$IN_VER, verify=$VERIFY) target=$TARGET gate-model=${GATE_MODEL:-<same>} =="
    for m in $(methods "$TARGET"); do
        for e in $(eps_set "$EPS_SET"); do
            local name; name="$(sel_name "$m" "$e" "$IN_VER" "$GATE")"
            local dir;  dir="$(find_selector "$name")"
            [[ -z "$dir" ]] && { echo "  skip eval $m $e: no model $name"; continue; }
            local trunk; trunk="${TRUNK_OVERRIDE:-$(trunk_for "$e")}"
            # ACE box cert: nat / pgd / cert_box (the verifier the released numbers use)
            if [[ "$VERIFY" == box || "$VERIFY" == all ]]; then
                local model; model="$(latest_net "$dir")"
                run "eval_${m}_${e}_box_${VER}" eval_ace.slurm \
                    --load-model "$model" \
                    --load-trunk-model "$trunk" \
                    --gate-type "$( [[ "$GATE" == ent ]] && echo entropy || echo net )" \
                    --gate-threshold "${TAU_OVERRIDE:-$(tau_grid "$GATE")}" \
                    --cert-domain box \
                    --eps "$(eps_of "$e")" \
                    --exp-id "$(eval_id "$m" "$e" "$GATE")" \
                    --exp-name "$(eval_name "$m" "$e" "$GATE" "$VER")"
            fi
            # alpha-CROWN: combine the standalone gate+branch labels (from the labels_eval stage).
            # ACE's own l_alpha is a no-op (alpha optimizer finds no params), so we do not use it.
            if [[ "$VERIFY" == alpha || "$VERIFY" == all ]]; then
                local npz="$RESULTS_DIR/dumps/labels_eval_${m}_${e}_${IN_VER}.npz"
                if [[ -f "$npz" ]]; then
                    local taus=()
                    [[ -n "$TAU_OVERRIDE" ]] && taus=(--taus "$TAU_OVERRIDE")
                    py "$SCRIPT_DIR/combine_labels_eval.py" --npz "$npz" "${taus[@]}" \
                        --out "$RESULTS_DIR/labels_eval_${m}_${e}_${IN_VER}.csv"
                else
                    echo "  skip alpha $m $e: no $npz (run: bash pipeline.sh labels_eval $TARGET --in_ver $IN_VER)"
                fi
            fi
        done
    done
}

stage_cross_gate() {
    echo "== cross_gate ($VER, in=$IN_VER, gate=$GATE_MODEL) target=$TARGET =="
    for m in $(methods "$TARGET"); do
        for e in $(eps_set "$EPS_SET"); do
            # branch from the target method's selector
            local bname; bname="$(sel_name "$m" "$e" "$IN_VER" "$GATE")"
            local bdir; bdir="$(find_selector "$bname")"
            [[ -z "$bdir" ]] && { echo "  skip cross_gate $m $e: no branch selector $bname"; continue; }
            # gate from the gate-model's selector
            local gname; gname="$(sel_name "$GATE_MODEL" "$e" "$IN_GATE_MODEL" "$GATE")"
            local gdir; gdir="$(find_selector "$gname")"
            [[ -z "$gdir" ]] && { echo "  skip cross_gate $m $e: no gate selector $gname"; continue; }
            local bnet; bnet="$(latest_net "$bdir")"
            local gnet; gnet="$(latest_net "$gdir")"
            [[ -z "$bnet" || -z "$gnet" ]] && { echo "  skip cross_gate $m $e: no net checkpoint"; continue; }
            # extract bare gate/branch from the combined selector checkpoints
            local xdir="$CONVERTED_DIR/xgate_${GATE_MODEL}_${m}_${e}_${VER}"
            py "$SCRIPT_DIR/extract_gate_branch.py" --ckpt "$gnet" --out-dir "$xdir/gate"
            py "$SCRIPT_DIR/extract_gate_branch.py" --ckpt "$bnet" --out-dir "$xdir/branch"
            local trunk; trunk="${TRUNK_OVERRIDE:-$(trunk_for "$e")}"
            local tau; tau="$(tau_grid "$GATE")"
            run "xgate_${m}_${e}_${VER}" eval_ace.slurm \
                --load-branch-model "$xdir/branch/branch.pt" \
                --load-gate-model "$xdir/gate/gate.pt" \
                --load-trunk-model "$trunk" \
                --gate-type "$( [[ "$GATE" == ent ]] && echo entropy || echo net )" \
                --gate-threshold "$tau" \
                --cert-domain box \
                --eps "$(eps_of "$e")" \
                --exp-id "$(eval_id "$m" "$e" "$GATE")" \
                --exp-name "$(eval_name "$m" "$e" "$GATE" "$VER")"
            # alpha verification: compose gate from one npz, branch from another
            local gate_npz="$RESULTS_DIR/dumps/labels_eval_${GATE_MODEL}_${e}_${IN_GATE_MODEL}.npz"
            local branch_npz="$RESULTS_DIR/dumps/labels_eval_${m}_${e}_${IN_VER}.npz"
            if [[ -f "$gate_npz" && -f "$branch_npz" ]]; then
                local taus=()
                [[ -n "$TAU_OVERRIDE" ]] && taus=(--taus "$TAU_OVERRIDE")
                mkdir -p "$RESULTS_DIR"
                py "$SCRIPT_DIR/combine_labels_eval.py" \
                    --npz "$branch_npz" --gate-npz "$gate_npz" "${taus[@]}" \
                    --out "$RESULTS_DIR/labels_eval_${m}_${e}_xgate-${GATE_MODEL}_${VER}.csv"
            else
                echo "  skip alpha cross_gate $m $e: missing npz (gate=$gate_npz branch=$branch_npz)"
            fi
        done
    done
}

stage_released() {
    echo "== released ($VER) target=$TARGET eps=$EPS_SET =="
    for m in $(rel_methods "$TARGET"); do
        for e in $(eps_set "$EPS_SET"); do
            local ck; ck="$(REL_MODEL "$m" "$e")"
            if [[ -z "$ck" ]]; then echo "  skip released $m $e: unknown released method"; continue; fi
            if [[ ! -f "$ck" && "$DRY" != "1" ]]; then echo "  skip released $m $e: missing $ck"; continue; fi
            local trunk; trunk="${TRUNK_OVERRIDE:-$(trunk_for "$e")}"
            run "rel_${m}_${e}_${VER}" eval_ace.slurm \
                --load-model "$ck" \
                --load-trunk-model "$trunk" \
                --gate-type net \
                --gate-threshold "${TAU_OVERRIDE:-$(tau_grid sel)}" \
                --cert-domain "$(rel_cert_domain "$m")" \
                --eps "$(eps_of "$e")" \
                --exp-id "$(rel_id "$m" "$e")" \
                --exp-name "rel_${m}_${e}_${VER}"
        done
    done
    echo "  submitted as exp-name rel_<method>_<eps>_$VER; run 'bash pipeline.sh aggregate --in_ver $VER' when done"
}

stage_roc_released() {
    echo "== roc_released ($VER) target=$TARGET eps=$EPS_SET =="
    for m in $(rel_methods "$TARGET"); do
        for e in $(eps_set "$EPS_SET"); do
            local ck; ck="$(REL_MODEL "$m" "$e")"
            if [[ -z "$ck" ]]; then echo "  skip roc_released $m $e: unknown released method"; continue; fi
            if [[ ! -f "$ck" && "$DRY" != "1" ]]; then echo "  skip roc_released $m $e: missing $ck"; continue; fi
            # ROC ground truth = "is the branch certifiable". Use box: it is the
            # cheap, standard choice (same as roc_released.pdf) and zono on a COLT
            # branch is far too slow/heavy (OOM + walltime limit). Override with
            # ROC_DOMAIN=zono if you really want the zono ROC (expect long runtimes).
            run "roc_rel_${m}_${e}_${VER}" gen_roc.slurm \
                --ckpt "$ck" --eps "$(eps_of "$e")" \
                --domain "${ROC_DOMAIN:-box}" \
                $([ "${ROC_DOMAIN:-box}" = zono ] && echo --ver-batch 8) \
                --output "$FIG_DIR/roc_released_${m}_${e}.pdf"
        done
    done
}

stage_aggregate() {
    echo "== aggregate (in=$IN_VER) =="
    local out="$RESULTS_DIR/agg_${IN_VER}.csv"
    py "$SCRIPT_DIR/aggregate_ace_results.py" \
        --results-dir "$ACE_DIR/models_new" \
        --output "$out" \
        --version "$IN_VER" \
        --labels-eval-dir "$RESULTS_DIR" \
        --kind "$KIND"
    echo "  -> $out"
}

# ---- dispatch ----------------------------------------------------------
case "$STAGE" in
    train_core) stage_train_core ;;
    eval_core)  stage_eval_core ;;
    train)      stage_train ;;
    convert)    stage_convert ;;
    labels)     stage_labels ;;
    selector)   stage_selector ;;
    roc)        stage_roc ;;
    cross_gate) stage_cross_gate ;;
    labels_eval)    stage_labels_eval ;;
    eval)       stage_eval ;;
    released)   stage_released ;;
    roc_released) stage_roc_released ;;
    aggregate)  stage_aggregate ;;
    all)        stage_train; wait_jobs; stage_convert; wait_jobs; stage_labels; wait_jobs; stage_selector; wait_jobs; stage_labels_eval; wait_jobs; stage_roc; stage_eval; wait_jobs; stage_aggregate ;;
    status)     bash "$SCRIPT_DIR/pipeline_status.sh" "$VER" ;;
    help|*)     sed -n '2,59p' "$0" ;;
esac

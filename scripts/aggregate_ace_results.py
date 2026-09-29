#!/usr/bin/env python3
"""Aggregate ACE evaluation results into a single table.

ACE writes one cert_log.csv per run into a timestamped directory:
    ACE/models_new/<dataset>/<exp_name>/<exp_id>/<net>_<eps>/<timestamp>/cert_log.csv

Each row is a sample; the per-run summary line printed at the end gives
nat_ok / adv_ok / cert_ok. This script walks a directory of runs and
collects the summary numbers into one CSV for the thesis table.

Usage:
    python aggregate_ace_results.py \
        --results-dir ~/research/ACE/models_new \
        --output ~/research/results/all_results.csv
"""
import argparse
import csv
import os
import re
import sys

# cert_log.csv comes in TWO layouts under the same filename:
#   stage="final"    cert_deepTrunk_net (deepTrunk_main.py) — the composed ACE net:
#     img_id;label;nat_ok;pgd_ok;ver_ok_<dom>...;nat_branch;pgd_branch;branch_<i>_p...;branch_<i>_adv_p...
#   stage="pre_gate"  diffAI_cert via get_gated_data (deepTrunk_main.py:270), written BEFORE gate
#     training to build the gate targets — BRANCH-ONLY numbers, no routing:
#     img_id;label;nat_ok;pgd_ok;ver_<dom>;ver_ok_<dom>;adv_threshold;cert_threshold_<dom>
# Never compare the two: "pre_gate" rows are a branch report card, not an ACE result.


def detect_cert_phase(header):
    # ACE writes cert_log.csv at two points during a run: an intermediate log during
    # gate training (has adv_threshold) and the definitive cert pass (has nat_branch).
    if "nat_branch" in header:
        return "final"
    if "adv_threshold" in header:
        return "pre_gate"
    return "unknown"


def read_summary(cert_log_path):
    """Extract per-sample flags from a cert_log.csv and return summary numbers."""
    with open(cert_log_path, "r") as f:
        reader = csv.reader(f, delimiter=";")
        try:
            header = next(reader)
        except StopIteration:
            return None  # empty cert_log.csv (crashed/partial run)
        header = [h.strip('"').strip() for h in header]
        rows = list(reader)

    if not header:
        return None

    # drop the trailing "total" summary row (nat_ok is a fraction there, not a flag)
    rows = [r for r in rows if not (r and r[0].strip().lower() in ("total",))]

    if not rows:
        return None

    # find column indices
    def col(name):
        return header.index(name)

    nat_ok = col("nat_ok")
    pgd_ok = [i for i, h in enumerate(header) if h.startswith("pgd_ok")]
    ver_ok = [i for i, h in enumerate(header) if h.startswith("ver_ok")]

    n = len(rows)
    nat = sum(int(r[nat_ok]) for r in rows if r[nat_ok].strip()) / n

    # pgd: take the first pgd_ok column (empirical robustness)
    pgd = 0.0
    if pgd_ok:
        pgd = sum(int(r[pgd_ok[0]]) for r in rows if r[pgd_ok[0]].strip()) / n

    # cert: any ver_ok_* column true counts as certified (multi-domain runs)
    cert = 0.0
    if ver_ok:
        cert = sum(
            1 for r in rows
            if any(r[i].strip() == "1" for i in ver_ok if i < len(r))
        ) / n

    return {"nat": nat, "pgd": pgd, "cert": cert, "n": n, "cert_phase": detect_cert_phase(header)}


def parse_run_dir(path):
    """Parse a timestamped run dir into (method, eps, tau) metadata.

    The directory layout is:
        <results-dir>/<dataset>/<exp_name>/<exp_id>/<net>_<eps>/<timestamp>
    where <net>_<eps> looks like 'efficientnet-b0_pre_0.00784'.
    """
    name = os.path.basename(path)
    parent = os.path.dirname(path)
    net_eps = os.path.basename(parent)
    exp_id = os.path.basename(os.path.dirname(parent))
    exp_name = os.path.basename(os.path.dirname(os.path.dirname(parent)))

    # tau/gate_type/train_mode are not in the path (they're in args.json)
    tau = None
    gate_type = None
    train_mode = None
    cert_domain = None
    args_path = os.path.join(path, "args.json")
    if os.path.exists(args_path):
        import json
        with open(args_path) as f:
            args = json.load(f)
        tau = args.get("gate_threshold")
        gate_type = args.get("gate_type")
        train_mode = args.get("train_mode")
        cd = args.get("cert_domain")
        if isinstance(cd, (list, tuple)):
            cd = "+".join(str(x) for x in cd)
        cert_domain = cd

    # net_eps: '<net>_<eps>' → eps is the last float token
    m = re.search(r"([0-9]+\.[0-9]+)$", net_eps)
    eps = float(m.group(1)) if m else None

    return {
        "run_dir": path,
        "exp_name": exp_name,
        "exp_id": exp_id,
        "net_eps": net_eps,
        "eps": eps,
        "tau": tau,
        "gate_type": gate_type,
        "train_mode": train_mode,
        "cert_domain": cert_domain,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True,
                    help="Root of ACE models_new (walks timestamped run dirs)")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--label", default=None,
                    help="Optional extra label column (e.g. 'sabr_2_255')")
    ap.add_argument("--version", default=None,
                    help="Filter by a version suffix in the exp-name. Use 'all' or omit for no filter.")
    ap.add_argument("--kind", default="all", choices=["all", "eval", "train"],
                    help="Filter by pipeline stage: eval (train_mode=cert) or train (selector/other).")
    ap.add_argument("--labels-eval-dir", default=None,
                    help="Directory containing labels_eval CSVs (from the compose step). "
                         "If set, cert_box/cert_crown/cert_alpha columns are added per tau.")
    args = ap.parse_args()

    version_filter = None
    if args.version and args.version != "all":
        version_filter = args.version

    rows_out = []
    for root, dirs, files in os.walk(args.results_dir):
        if "cert_log.csv" not in files:
            continue
        meta = parse_run_dir(root)

        # version filter: exp-name must end with _<version>
        if version_filter and not meta["exp_name"].endswith(f"_{version_filter}"):
            continue

        # kind: our pipeline stage. Prefer the run's own train_mode from args.json
        # (cert = evaluation, anything else = a training run such as selector training);
        # fall back to the _eval_ naming only if args.json is missing.
        tm = meta.get("train_mode")
        if tm == "cert":
            kind = "eval"
        elif tm:
            kind = "train"
        else:
            kind = "eval" if "_eval_" in meta["exp_name"] else "train"
        if args.kind and args.kind != "all" and kind != args.kind:
            continue

        cert_log_path = os.path.join(root, "cert_log.csv")
        summary = read_summary(cert_log_path)
        if summary is None:
            print(f"  [warn] skipping {root}: cert_log.csv empty or no usable data",
                  file=sys.stderr)
            continue
        row = {
            "run_dir": meta["run_dir"],
            "exp_name": meta["exp_name"],
            "exp_id": meta["exp_id"],
            "eps": meta["eps"],
            "tau": meta["tau"],
            "gate_type": meta["gate_type"],
            "kind": kind,
            "cert_phase": summary["cert_phase"],
            "nat_acc": summary["nat"],
            "pgd_acc": summary["pgd"],
            "box_cert_acc": summary["cert"],
            "n_samples": summary["n"],
        }
        if args.label:
            row["label"] = args.label
        rows_out.append(row)

    if not rows_out:
        print(f"No cert_log.csv found under {args.results_dir}")
        sys.exit(1)

    # -- labels_eval merge: attach cert_box/cert_crown/cert_alpha per tau -----------------
    le_by_exp_tau = {}  # (exp_name, tau) -> {routing_box, branch_ver_box, cert_box, ...}
    if args.labels_eval_dir:
        _methods = ("mtl_ibp", "crown_ibp", "ibp", "sabr")  # longer prefixes first
        _epses = ("2_255", "8_255")
        _prefixes = [f"{m}_{e}" for m in _methods for e in _epses]
        n_files = 0
        for fname in os.listdir(args.labels_eval_dir):
            if not fname.startswith("labels_eval_") or not fname.endswith(".csv"):
                continue
            stem = fname[len("labels_eval_"):-len(".csv")]
            # Accept the cross-gate naming too:
            #   xgate-<gate>_<method>_<eps>_<ver> -> <method>_<eps>_xgate-<gate>_<ver>
            # (the cross-gate eval exp-name is <m>_<eps>_xgate-<gate>_sel_eval_<ver>,
            #  whose selector name is <m>_<eps>_xgate-<gate>_<ver>).
            mo = re.match(
                r"xgate-(?P<gate>.+?)_(?P<m>mtl_ibp|crown_ibp|ibp|sabr)_(?P<eps>2_255|8_255)_(?P<ver>.+)$",
                stem)
            if mo:
                stem = f"{mo['m']}_{mo['eps']}_xgate-{mo['gate']}_{mo['ver']}"
            exp_name = None
            for p in _prefixes:
                if stem.startswith(p + "_"):
                    exp_name = stem
                    break
            if exp_name is None:
                continue
            if version_filter and not exp_name.endswith(f"_{version_filter}"):
                continue
            path = os.path.join(args.labels_eval_dir, fname)
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    try:
                        tau = float(row["tau"])
                        v = row["verifier"]
                    except (KeyError, ValueError):
                        continue
                    key = (exp_name, tau)
                    if key not in le_by_exp_tau:
                        le_by_exp_tau[key] = {}
                    le_by_exp_tau[key][f"cert_{v}"] = float(row.get("cert", "nan"))
                    le_by_exp_tau[key][f"routing_{v}"] = float(row.get("routing", "nan"))
                    le_by_exp_tau[key][f"branch_ver_{v}"] = float(row.get("branch_ver", "nan"))
            n_files += 1
        if le_by_exp_tau:
            print(f"  [labels_eval] merged {n_files} files ({len(le_by_exp_tau)} exp_name/tau pairs)")
        else:
            print(f"  [labels_eval] no matching files in {args.labels_eval_dir}")

    def sel_name_from_eval(en):
        """eval_name '<m>_<eps>_sel_eval_<ver>' -> sel_name '<m>_<eps>_<ver>'."""
        return en.replace("_sel_eval_", "_")

    # attach cert columns to aggregate rows
    has_le = bool(le_by_exp_tau)
    le_cert_keys = ["alpha_cert_acc"]
    if has_le:
        for row in rows_out:
            key = (sel_name_from_eval(row["exp_name"]), row["tau"])
            le = le_by_exp_tau.get(key, {})
            row["alpha_cert_acc"] = le.get("cert_alpha", float("nan"))
            for k in ["routing_box", "routing_crown", "routing_alpha",
                      "branch_ver_box", "branch_ver_crown", "branch_ver_alpha"]:
                row[k] = le.get(k, float("nan"))

    fields = ["label", "exp_name", "exp_id", "eps", "tau", "gate_type", "kind", "cert_phase",
              "nat_acc", "pgd_acc", "box_cert_acc"]
    if has_le:
        fields += ["alpha_cert_acc"]
    fields += ["n_samples", "run_dir"]
    if has_le:
        fields += ["routing_box", "routing_crown", "routing_alpha",
                   "branch_ver_box", "branch_ver_crown", "branch_ver_alpha"]
    fields = [f for f in fields if any(f in r for r in rows_out)]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} runs to {args.output}")
    for r in sorted(rows_out, key=lambda x: (str(x.get("eps")), str(x.get("tau")))):
        alpha = r.get("alpha_cert_acc")
        alpha_s = f"{alpha:.4f}" if isinstance(alpha, float) else "-"
        print(f"  eps={r['eps']} tau={r['tau']} gate={r['gate_type']} kind={r['kind']} "
              f"phase={r['cert_phase']} "
              f"nat={r['nat_acc']:.4f} pgd={r['pgd_acc']:.4f} box={r['box_cert_acc']:.4f} alpha={alpha_s}")


if __name__ == "__main__":
    main()

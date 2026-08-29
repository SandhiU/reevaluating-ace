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

# cert_log.csv column layout (from deepTrunk_main.py cert_deepTrunk_net):
# img_id; label; nat_ok; pgd_ok; ver_ok_<domain>...; nat_branch; pgd_branch; ...
# The "ver_ok_*" column(s) hold the certified flag; nat_ok the natural one.


def read_summary(cert_log_path):
    """Extract per-sample flags from a cert_log.csv and return summary numbers."""
    with open(cert_log_path, "r") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        # sanitize header (fields may be quoted)
        header = [h.strip('"').strip() for h in header]
        rows = list(reader)

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

    return {"nat": nat, "pgd": pgd, "cert": cert, "n": n}


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

    # tau is not in the path (it's in args.json) — try to read it
    tau = None
    args_path = os.path.join(path, "args.json")
    if os.path.exists(args_path):
        import json
        with open(args_path) as f:
            args = json.load(f)
        tau = args.get("gate_threshold")
        gate_type = args.get("gate_type")

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
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True,
                    help="Root of ACE models_new (walks timestamped run dirs)")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--label", default=None,
                    help="Optional extra label column (e.g. 'sabr_2_255')")
    args = ap.parse_args()

    rows_out = []
    for root, dirs, files in os.walk(args.results_dir):
        if "cert_log.csv" not in files:
            continue
        meta = parse_run_dir(root)
        summary = read_summary(os.path.join(root, "cert_log.csv"))
        if summary is None:
            continue
        row = {
            "run_dir": meta["run_dir"],
            "exp_name": meta["exp_name"],
            "exp_id": meta["exp_id"],
            "eps": meta["eps"],
            "tau": meta["tau"],
            "gate_type": meta["gate_type"],
            "nat_acc": summary["nat"],
            "pgd_acc": summary["pgd"],
            "cert_acc": summary["cert"],
            "n_samples": summary["n"],
        }
        if args.label:
            row["label"] = args.label
        rows_out.append(row)

    if not rows_out:
        print(f"No cert_log.csv found under {args.results_dir}")
        sys.exit(1)

    fields = ["label", "exp_name", "exp_id", "eps", "tau", "gate_type",
              "nat_acc", "pgd_acc", "cert_acc", "n_samples", "run_dir"]
    fields = [f for f in fields if any(f in r for r in rows_out)]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} runs to {args.output}")
    for r in sorted(rows_out, key=lambda x: (str(x.get("eps")), str(x.get("tau")))):
        print(f"  eps={r['eps']} tau={r['tau']} gate={r['gate_type']} "
              f"nat={r['nat_acc']:.4f} pgd={r['pgd_acc']:.4f} cert={r['cert_acc']:.4f}")


if __name__ == "__main__":
    main()

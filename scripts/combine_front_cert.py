#!/usr/bin/env python3
"""Combine gate-routing margins with branch cert margins into a composed certified-accuracy column.

Why this design: ACE's composed certification factorises as

    cert( tau ) = P( gate provably routes at tau  AND  branch certifies )

where the gate side is verifier-independent of the branch (routing stays on ACE's cheap box
verifier) and the branch side does not depend on tau at all. So we compute the two sides once
and get every tau for free:

  * gate side   : `analyze_gate_margins.py --dump-npz`  (cheap, box bounds, one job per branch)
  * branch side : `gen_labels.py` on the TEST split (no --train), one job per branch.
                  Its `label` column is exactly "the branch verifies under alpha-CROWN", sample
                  aligned by CIFAR-10 test position.

Both are on the unshuffled test order, so they align by index. The natural accuracy still comes
from ACE's own eval (cert_log / agg CSV), because it involves the core network and cannot be
recomputed from these two sides.

Usage:
  python combine_front_cert.py \
      --gate-npz 'dumps/*.npz' \
      --alpha-csv '~/research/labels_test/labels_alpha_crown_*.csv' \
      --agg ~/research/scripts/results/agg_sel_eval_alpha_v3.csv \
      --out results/agg_sel_eval_alpha_v3_with_acrown.csv

Per branch it prints and writes, for every tau on the grid:
  cert_box_recomputed  (= P(lb>tau AND box verifies), should match the agg cert_acc)
  cert_alpha           (= P(lb>tau AND alpha-CROWN verifies))
  cert_combined        (= P(lb>tau AND (box OR alpha verifies)), always >= each individually)
  nat                  (from the agg CSV, verifier-independent)
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np

TAU_DEFAULT = [-2.0, -1.5, -1.0, -0.7, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]


def _expand(p):
    """Expand ~ and $VARS. Single-quoting the glob in the shell (`'$HOME/dumps/*.npz'`)
    suppresses $HOME expansion, so python must do it (18 Sep: that is why
    `no gate npz matched $HOME/research/dumps/*.npz` appeared literally)."""
    return os.path.expandvars(os.path.expanduser(p))


def read_alpha(path):
    """index -> bool label, from a gen_labels CSV."""
    lut = {}
    with open(path) as fh:
        rd = csv.DictReader(fh)
        if "index" not in rd.fieldnames or "label" not in rd.fieldnames:
            raise SystemExit(f"{path}: needs 'index' and 'label' columns")
        for row in rd:
            lut[int(row["index"])] = str(row["label"]).strip().lower() in ("1", "true", "yes")
    return lut


def read_agg(path):
    """(branch_prefix, tau) -> (nat, cert_box) from an aggregate eval CSV."""
    if not path or not os.path.exists(path):
        return {}
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            name = row.get("exp_name", "")
            try:
                key = (name, float(row["tau"]))
                out[key] = (float(row["nat_acc"]), float(row["cert_acc"]))
            except (KeyError, ValueError):
                continue
    return out


def describe_agg(path):
    """Print what we actually got from --agg, so a silent nan is diagnosable in one run."""
    print(f"[agg] file   : {path}")
    print(f"[agg] exists : {os.path.exists(path)}")
    if not os.path.exists(path):
        return
    with open(path) as fh:
        rd = csv.DictReader(fh)
        names, taus, n = set(), set(), 0
        for row in rd:
            n += 1
            names.add(row.get("exp_name", ""))
            try:
                taus.add(float(row["tau"]))
            except (KeyError, ValueError, TypeError):
                pass
    print(f"[agg] rows   : {n}")
    print(f"[agg] names  : {sorted(names)[:10]}")
    print(f"[agg] taus   : {sorted(taus)}")


def branch_of(path):
    return os.path.basename(path).split(".")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-npz", required=True, help="glob of analyze_gate_margins dumps")
    ap.add_argument("--alpha-csv", required=True, help="glob of test-split alpha-CROWN label CSVs")
    ap.add_argument("--agg", default=None, help="aggregate eval CSV, for the nat column")
    ap.add_argument("--taus", default=",".join(str(t) for t in TAU_DEFAULT))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    taus = [float(t) for t in a.taus.split(",")]
    npzs = sorted(glob.glob(_expand(a.gate_npz)))
    csvs = sorted(glob.glob(_expand(a.alpha_csv)))
    if not npzs:
        sys.exit(f"no gate npz matched {a.gate_npz}")
    if not csvs:
        sys.exit(f"no alpha csv matched {a.alpha_csv}")

    alpha_by_branch = {}
    for f in csvs:
        b = branch_of(f)
        for cand in (b, re.sub(r"^labels_alpha_crown_", "", b)):
            alpha_by_branch[cand] = f
    agg = read_agg(_expand(a.agg) if a.agg else None)
    if a.agg:
        describe_agg(_expand(a.agg))
        print(f"[agg] usable keys: {len(agg)} (need exp_name/tau/nat_acc/cert_acc)")
    else:
        print("[warn] no --agg given: nat/agg_box columns will be nan")

    rows = []
    warned = {}
    for npz in npzs:
        b = branch_of(npz)
        d = np.load(npz)
        lb, box_ver, idx = d["lb"], d["box_ver"].astype(bool), d["idx"]
        n = len(lb)

        ckey = None
        for cand in (b, b + "_sel_eval_alpha_v3", b.replace("_2_255", "_2_255_sel_eval_alpha_v3")):
            if cand in alpha_by_branch:
                ckey = cand
                break
        if ckey is None:
            print(f"!! {b}: no alpha CSV matched (have: {sorted(alpha_by_branch)}), skipping")
            continue
        lut = read_alpha(alpha_by_branch[ckey])
        missing = [int(i) for i in idx if int(i) not in lut]
        if missing:
            print(f"!! {b}: {len(missing)} of {n} test indices missing from "
                  f"{alpha_by_branch[ckey]} (e.g. {missing[:5]}); skipping")
            continue
        alpha = np.array([lut[int(i)] for i in idx], dtype=bool)

        print(f"\n=== {b}  (n={n}, gate dump eps={float(d['eps']):.5f}, "
              f"alpha csv={os.path.basename(alpha_by_branch[ckey])}) ===")
        print(f"  branch box-ver rate      : {box_ver.mean():.4f}")
        print(f"  branch alpha-CROWN rate  : {alpha.mean():.4f}")
        print(f"  alpha >= box (must hold) : {'OK' if alpha.mean() >= box_ver.mean() else 'VIOLATED'}")
        print(f"  {'tau':>5} {'nat':>7} {'cert_box':>9} {'agg_box':>8} {'d':>7} {'cert_alpha':>11} {'cert_combined':>14}")
        for tau in taus:
            routed = lb > tau
            cb = float((routed & box_ver).mean())
            ca = float((routed & alpha).mean())
            cc = float((routed & (box_ver | alpha)).mean())
            nat, agg_cb = agg.get((b + "_sel_eval_alpha_v3", tau), (float("nan"), float("nan")))
            if np.isnan(nat):
                for (name, t), (na, ce) in agg.items():
                    if name.startswith(b) and t == tau:
                        nat, agg_cb = na, ce
                        break
            if np.isnan(nat) and not warned.get(b):
                warned[b] = True
                near = sorted({n for (n, _t) in agg if b.split("_2_255")[0] in n or b in n})[:6]
                print(f"  [warn] no agg row matched branch '{b}' at tau={tau}; "
                      f"closest agg names: {near}")
            dlt = cb - agg_cb
            print(f"  {tau:5.1f} {nat:7.4f} {cb:9.4f} {agg_cb:8.4f} {dlt:7.4f} {ca:11.4f} {cc:14.4f}")
            rows.append({"branch": b, "tau": tau, "nat_acc": nat,
                         "cert_box_recomputed": cb, "cert_box_agg": agg_cb,
                         "cert_alpha": ca, "cert_combined": cc, "n_samples": n,
                         "route_rate": float(routed.mean())})

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.out} ({len(rows)} rows)")
    print("\nchecks:  (1) 'd' should be ~0 if the gate dump used the same sample count as the agg "
          "eval (use --samples 10000);\n         (2) alpha rate must be >= box rate on every branch "
          "(tighter verifier certifies more);\n         (3) monotonicity cert_alpha >= cert_box at "
          "every tau.")


if __name__ == "__main__":
    main()

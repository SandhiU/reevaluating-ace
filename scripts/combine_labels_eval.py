#!/usr/bin/env python3
"""Compose gate routing and branch verification (from a gen_margins npz) into per-tau cert.

    cert(tau, V) = P( gate_lb_V > tau  AND  branch_margin_V > 0 )

for V in {box, crown, alpha}, so the three verifiers sit side by side at every tau.

Usage:
  python combine_labels_eval.py --npz results/dumps/margins_ibp_2_255.npz \
      --out results/margins_ibp_2_255.csv

nat is verifier-independent (it needs the full trunk+gate+branch model, which this
standalone verifier never builds), so it cannot be read off the npz. Pass --agg (the
aggregate stage's agg_<ver>.csv) and --exp-name to attach the composed nat per tau from
ACE's own box eval, so the output table is nat + routing + branch + cert in one file.
"""
import argparse
import csv
import os

import numpy as np

TAU = [-2.0, -1.5, -1.0, -0.7, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True,
                    help="labels_eval npz (gate + branch from the same selector). Used for both "
                         "gate and branch unless --gate-npz or --branch-npz override.")
    ap.add_argument("--gate-npz", default=None,
                    help="separate npz for the gate (cross-gate eval). Takes gate_lb_* from "
                         "this file and branch_margin_* from --npz (or --branch-npz).")
    ap.add_argument("--branch-npz", default=None,
                    help="separate npz for the branch (cross-gate eval). Takes branch_margin_* "
                         "from this file and gate_lb_* from --npz (or --gate-npz).")
    ap.add_argument("--taus", default=",".join(str(t) for t in TAU))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d_gate = np.load(a.gate_npz or a.npz)
    d_branch = np.load(a.branch_npz or a.npz)
    taus = [float(t) for t in a.taus.split(",")]
    n = len(d_gate["idx"])
    if len(d_branch["idx"]) != n:
        sys.exit(f"gate and branch npz have different sample counts: "
                 f"{len(d_gate['idx'])} vs {len(d_branch['idx'])}")

    if a.gate_npz or a.branch_npz:
        print(f"  cross-gate: gate={os.path.basename(a.gate_npz or a.npz)} "
              f"branch={os.path.basename(a.branch_npz or a.npz)}")
    print(f"=== {os.path.basename(a.npz)}  (n={n}, eps={float(d_gate['eps']):.5f}) ===")
    print(f"  {'verifier':<8} {'tau':>6} {'routing':>9} {'branch':>8} {'cert':>8}")
    rows = []
    for v in ("box", "crown", "alpha"):
        glb = d_gate[f"gate_lb_{v}"]
        bm = d_branch[f"branch_margin_{v}"] > 0
        for t in taus:
            r = glb > t
            row = {"verifier": v, "tau": t,
                   "routing": float(r.mean()),
                   "branch_ver": float(bm.mean()),
                   "cert": float((r & bm).mean())}
            rows.append(row)
    for v in ("box", "crown", "alpha"):
        sub = [r for r in rows if r["verifier"] == v]
        # print a couple of informative taus only
        for r in sub:
            if r["tau"] in (0.0, -1.0):
                print(f"  {v:<8} {r['tau']:>6.1f} {r['routing']:>9.4f} {r['branch_ver']:>8.4f} {r['cert']:>8.4f}")

    ok = True
    for t in taus:
        al = next(r["cert"] for r in rows if r["verifier"] == "alpha" and r["tau"] == t)
        bx = next(r["cert"] for r in rows if r["verifier"] == "box" and r["tau"] == t)
        # alpha-CROWN (optimized intermediates) must be >= box (IBP). The crown column
        # is CROWN-IBP (backward with IBP intermediates) and can legitimately sit below
        # box; that's a known auto_LiRPA artifact, not a violation.
        if not (al >= bx - 1e-9):
            ok = False
    print(f"\n  monotone  alpha >= box at every tau : {'OK' if ok else 'VIOLATED'}")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        fields = ["verifier", "tau", "routing", "branch_ver", "cert"]
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
        print(f"  wrote {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

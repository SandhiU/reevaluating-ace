#!/usr/bin/env python3
"""Compose gate routing and branch verification (from a gen_margins npz) into per-tau cert.

    cert(tau, V) = P( gate_lb_V > tau  AND  branch_margin_V > 0 )

for V in {box, crown, alpha}, so the three verifiers sit side by side at every tau.

Usage:
  python combine_margins.py --npz results/dumps/margins_ibp_2_255_v5.npz \
      --out results/margins_ibp_2_255_v5.csv
"""
import argparse
import csv
import os

import numpy as np

TAU = [-2.0, -1.5, -1.0, -0.7, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--taus", default=",".join(str(t) for t in TAU))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = np.load(a.npz)
    taus = [float(t) for t in a.taus.split(",")]
    n = len(d["idx"])
    print(f"=== {os.path.basename(a.npz)}  (n={n}, eps={float(d['eps']):.5f}) ===")
    print(f"  {'verifier':<8} {'tau':>6} {'routing':>9} {'branch':>8} {'cert':>8}")
    rows = []
    for v in ("box", "crown", "alpha"):
        glb = d[f"gate_lb_{v}"]
        bm = d[f"branch_margin_{v}"] > 0
        for t in taus:
            r = glb > t
            rows.append({"verifier": v, "tau": t,
                         "routing": float(r.mean()),
                         "branch_ver": float(bm.mean()),
                         "cert": float((r & bm).mean())})
    for v in ("box", "crown", "alpha"):
        sub = [r for r in rows if r["verifier"] == v]
        # print a couple of informative taus only
        for r in sub:
            if r["tau"] in (0.0, -1.0):
                print(f"  {v:<8} {r['tau']:>6.1f} {r['routing']:>9.4f} {r['branch_ver']:>8.4f} {r['cert']:>8.4f}")

    ok = True
    for t in taus:
        cw = next(r["cert"] for r in rows if r["verifier"] == "crown" and r["tau"] == t)
        al = next(r["cert"] for r in rows if r["verifier"] == "alpha" and r["tau"] == t)
        bx = next(r["cert"] for r in rows if r["verifier"] == "box" and r["tau"] == t)
        if not (al >= cw - 1e-9 and cw >= bx - 1e-9):
            ok = False
    print(f"\n  monotone  box <= crown <= alpha at every tau : {'OK' if ok else 'VIOLATED'}")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["verifier", "tau", "routing", "branch_ver", "cert"])
            w.writeheader(); w.writerows(rows)
        print(f"  wrote {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

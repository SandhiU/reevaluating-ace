#!/usr/bin/env python3
"""Extract bare gate and branch state dicts from an ACE selector checkpoint.

ACE's dTNet stores gate and branch as registered sub-modules:
    gateNet_0.blocks.layers.<i>.weight ...
    branchNet_0.blocks.layers.<i>.weight ...

eval_ace.slurm's --load-gate-model / --load-branch-model expect bare state dicts
with keys like blocks.layers.<i>.weight (no gateNet_0 prefix). This script extracts
them.

Usage:
    python extract_gate_branch.py --ckpt <selector net_*.pt> --out-dir <dir>
    # writes <dir>/gate.pt and <dir>/branch.pt
"""
import argparse
import os
import sys
import torch


def extract_prefix(sd, prefix):
    """Extract keys starting with prefix, strip it."""
    out = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="Selector checkpoint (net_*.pt)")
    ap.add_argument("--out-dir", required=True, help="Output directory for gate.pt and branch.pt")
    ap.add_argument("--exit-idx", type=int, default=0,
                    help="Exit index (default 0, matching ACE's first branch)")
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu")

    gate_prefix = f"gateNet_{args.exit_idx}."
    branch_prefix = f"branchNet_{args.exit_idx}."

    gate_sd = extract_prefix(sd, gate_prefix)
    branch_sd = extract_prefix(sd, branch_prefix)

    if not gate_sd:
        sys.exit(f"No keys with prefix '{gate_prefix}' in {args.ckpt}")
    if not branch_sd:
        sys.exit(f"No keys with prefix '{branch_prefix}' in {args.ckpt}")

    os.makedirs(args.out_dir, exist_ok=True)
    gate_path = os.path.join(args.out_dir, "gate.pt")
    branch_path = os.path.join(args.out_dir, "branch.pt")
    torch.save(gate_sd, gate_path)
    torch.save(branch_sd, branch_path)
    print(f"Extracted {len(gate_sd)} gate keys -> {gate_path}")
    print(f"Extracted {len(branch_sd)} branch keys -> {branch_path}")


if __name__ == "__main__":
    main()

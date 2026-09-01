#!/usr/bin/env python3
"""Verify a CTRAIN-format branch with auto_LiRPA (IBP / CROWN / alpha-CROWN).

auto_LiRPA is vendored in the ACE environment (../auto_LiRPA_Repo, commit 9d100ec).
This script wraps the branch (a plain nn.Sequential in CTRAIN format) and reports
certified accuracy under each method.

Motivation: ACE's own box (IBP-style) bounds are sound but loose. CROWN and
alpha-CROWN bounds are sound and tighter, so for the same gate routing the
composed certified accuracy can only improve. The gate routing itself stays on
ACE's cheap box verifier; only the branch verification is upgraded.

Usage (HPC, from ~/research/ACE, ace env):
  python ~/research/scripts/verify_branch_autolirpa.py \
      --ckpt ~/research/runs/ibp_eps0.00784314/final_model_state.pt \
      --eps 0.00784313725 --samples 2000 --methods ibp,crown,alpha-crown

Notes:
  - The CTRAIN checkpoint is nn.Sequential with indices 0,2,4,7,9 (C3 arch).
  - Normalization is prepended INSIDE the module (same mean/sigma as ACE), so the input
    ball is the raw-pixel eps ball, matching the ACE frame exactly.
  - Sanity check: auto_LiRPA-IBP should reproduce ACE's box certified accuracy;
    crown should be at least as high, alpha-crown at least crown.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ACE + vendored auto_LiRPA repos; resolve explicitly so the script works from any cwd.
_ACE_DIR = os.path.expanduser("~/research/ACE")
if os.path.isdir(_ACE_DIR) and _ACE_DIR not in sys.path:
    sys.path.insert(0, _ACE_DIR)
_AUTOLIRPA_DIR = os.path.expanduser("~/research/auto_LiRPA_Repo")
if os.path.isdir(_AUTOLIRPA_DIR) and _AUTOLIRPA_DIR not in sys.path:
    sys.path.insert(0, _AUTOLIRPA_DIR)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm  # noqa: E402

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)


class NormalizedBranch(nn.Module):
    """C3 branch with in-net normalization (matches ACE's Normalization block)."""

    def __init__(self, in_channel=3):
        super().__init__()
        self.register_buffer("mean", MEAN)
        self.register_buffer("std", STD)
        self.body = nn.Sequential(
            nn.Conv2d(in_channel, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 250),
            nn.ReLU(),
            nn.Linear(250, 10),
        )

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.body(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eps", type=float, default=0.00784313725)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--methods", default="ibp,crown,alpha-crown")
    ap.add_argument("--data-root", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # data (CIFAR-10 test)
    from loaders import get_loaders
    sys_argv = sys.argv[:]
    sys.argv = [sys_argv[0], "--net", "None"]
    from args_factory import get_args
    try:
        args = get_args()
    finally:
        sys.argv = sys_argv
    args.dataset = "cifar10"
    args.net = "None"
    args.test_batch = 64
    args.num_workers = 4
    args.test_set = "test"
    if a.data_root is not None:
        args.data_root = a.data_root
    _, _, test_loader, _, _, _ = get_loaders(args)

    model = NormalizedBranch().to(dev)
    sd = torch.load(a.ckpt, map_location=dev)
    # CTRAIN Sequential keys: 0/2/4/7/9 -> body layers; strip nothing (plain dict)
    model.body.load_state_dict(sd, strict=False)

    dummy = torch.zeros(1, 3, 32, 32).to(dev)
    bounded = BoundedModule(model, dummy)

    for method in [m.strip() for m in a.methods.split(",") if m.strip()]:
        certified = 0.0
        total = 0
        for x, y in test_loader:
            if total >= a.samples:
                break
            b = x.size(0)
            total += b
            x_b = x.to(dev)
            x_b = BoundedTensor(x_b, PerturbationLpNorm(norm=np.inf, eps=a.eps))
            with torch.no_grad():
                lb, ub = bounded.compute_bounds(x=x_b, method=method)
            # margin of true class: gather lb[y] - max_j!=y ub[j]
            y = y.to(dev)
            lb_y = lb.gather(1, y.view(-1, 1)).squeeze(1)
            ub_others = ub.clone()
            ub_others.scatter_(1, y.view(-1, 1), float("-inf"))
            ub_max = ub_others.max(1).values
            ok = (lb_y - ub_max > 0).float().sum().item()
            certified += ok
        print(f"{method:12s}: certified accuracy {certified / total:.4f} "
              f"({int(certified)}/{total}, eps={a.eps})")


if __name__ == "__main__":
    main()

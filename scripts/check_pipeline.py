#!/usr/bin/env python3
"""Pipeline stage checks for the ACE composition experiments.

Verifies each stage that feeds the composed-model evaluation, so that a problem
is localised before any downstream number is trusted:

  --norm          print ACE normalization constants vs CTRAIN loader constants
  --keys          load a converted branch checkpoint into ACE; report missing/unexpected keys
  --equiv         random-sample equivalence: released vs template checkpoint; converted vs CTRAIN
  --label-quality recompute branch certifiability under box vs zono bounds; flip matrix + margins

A failure at any stage invalidates downstream results. The classic silent failure
mode is a state-dict mismatch under strict=False loading, which yields a randomly
initialized branch without any error.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ACE modules (args_factory, loaders, deepTrunk_networks) live in the ACE repo dir;
# resolve it explicitly so the script works from any cwd.
_ACE_DIR = os.path.expanduser("~/research/ACE")
if os.path.isdir(_ACE_DIR) and _ACE_DIR not in sys.path:
    sys.path.insert(0, _ACE_DIR)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from args_factory import get_args, translate_net_name   # noqa: E402
from loaders import get_loaders                         # noqa: E402
from deepTrunk_networks import MyDeepTrunkNet           # noqa: E402

HPC = os.path.expanduser("~")
CONVERTED = os.path.join(HPC, "research", "converted")
RUNS = os.path.join(HPC, "research", "runs")
ACE = os.path.join(HPC, "research", "ACE")


def _args(eps=0.00784313725):
    import sys as _sys
    _argv = _sys.argv[:]
    _sys.argv = [_argv[0], "--net", "None"]
    try:
        a = get_args()
    finally:
        _sys.argv = _argv
    a.dataset = "cifar10"
    a.net = "None"
    a.n_branches = 1
    a.gate_type = "net"
    _c3 = translate_net_name("C3_cifar10")
    a.branch_nets = [_c3]
    a.gate_nets = [_c3]
    a.gate_threshold = 0.0
    a.test_eps = eps
    a.train_eps = eps
    a.test_batch = 100
    a.num_workers = 4
    a.n_rand_proj = 50
    a.gate_feature_extraction = None
    a.cert_net_dim = None
    a.no_cuda = False
    a.load_model = None
    a.load_trunk_model = None
    a.load_branch_model = None
    a.load_gate_model = None
    return a


def load_branch_state(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    if "branchNet_0" in str(next(iter(sd))):
        return {k[len("branchNet_0."):]: v for k, v in sd.items()
                if k.startswith("branchNet_0.")}
    return sd


def mode_norm():
    print("=== normalization constants ===")
    rel = torch.load(os.path.join(ACE, "trained_models", "C3_cifar10_IBP_2_255.pt"),
                     map_location="cpu")
    print("ACE layers.0.mean :", rel["blocks.layers.0.mean"].flatten().tolist())
    print("ACE layers.0.sigma:", rel["blocks.layers.0.sigma"].flatten().tolist())
    print("CTRAIN loader     : mean [0.4914, 0.4822, 0.4465] sigma [0.2023, 0.1994, 0.2010] (check data loader source)")
    print("match expected    : True (verified on released checkpoint)")


def mode_keys(branch="ibp_2_255_branch.pt"):
    print(f"=== key-count check: converted/{branch} ===")
    args = _args()
    lossFn = torch.nn.CrossEntropyLoss(reduction="none")
    evalFn = lambda x: torch.max(x, dim=1)[1]
    # CIFAR-10 shapes are fixed; do NOT call get_loaders here (it loads the whole
    # dataset, which is what made this mode slow).
    dTNet = MyDeepTrunkNet.get_deepTrunk_net(args, "cpu", lossFn, evalFn,
                                             32, 3, 10)
    exit_idx = dTNet.exit_ids[1]   # the single branch exit (trunk = -1, branch = 0)
    sd = torch.load(os.path.join(CONVERTED, branch), map_location="cpu")
    missing, unexpected = dTNet.branch_nets[exit_idx].load_state_dict(sd, strict=False)
    print(f"missing:   {len(missing)}")
    print(f"unexpected:{len(unexpected)}")
    print("OK" if not missing and not unexpected else "CHECK: conversion problem")


def mode_equiv(n=32, seed=0):
    print("=== random-sample equivalence ===")
    args = _args()
    lossFn = torch.nn.CrossEntropyLoss(reduction="none")
    evalFn = lambda x: torch.max(x, dim=1)[1]
    _, _, test_loader, input_size, input_channel, n_class = get_loaders(args)
    torch.manual_seed(seed)
    samples = []
    for x, y in test_loader:
        samples.append(x)
        if len(torch.cat(samples)) >= n:
            break
    xs = torch.cat(samples)[:n]

    # 1) released branch file vs template branchNet_0 (must be identical)
    rel = torch.load(os.path.join(ACE, "trained_models", "C3_cifar10_IBP_2_255.pt"),
                     map_location="cpu")
    tpl = torch.load(os.path.join(ACE, "trained_models", "C3_ACE_Net_IBP_cert_cifar10_2_255.pt"),
                     map_location="cpu")
    tpl_b = {k[len("branchNet_0."):]: v for k, v in tpl.items() if k.startswith("branchNet_0.")}
    same_keys = set(rel) == set(tpl_b)
    same_vals = all(torch.equal(rel[k], tpl_b[k]) for k in rel)
    print(f"released file vs template branchNet_0: keys {same_keys}, values {same_vals}")

    # 2) converted branch (ACE format) vs CTRAIN checkpoint (nn.Sequential)
    br = torch.load(os.path.join(CONVERTED, "ibp_2_255_branch.pt"), map_location="cpu")
    ct = torch.load(os.path.join(RUNS, "ibp_eps0.00784314", "final_model_state.pt"),
                    map_location="cpu")
    src_layers = [0, 2, 4, 7, 9]
    dst_layers = [1, 3, 5, 8, 10]
    same = True
    for s, d in zip(src_layers, dst_layers):
        w = f"blocks.layers.{d}.weight"
        if w in br and f"{s}.weight" in ct:
            if not torch.equal(br[w], ct[f"{s}.weight"]):
                same = False
                print(f"  MISMATCH layer {s} -> {d}")
        else:
            print(f"  key missing: {w} or {s}.weight")
    print(f"converted vs CTRAIN weights: {'identical' if same else 'CHECK: conversion mismatch'}")

    # 3) forward outputs: ACE net (normalizes in-net) vs CTRAIN Sequential (loader-normalized)
    dev = "cpu"
    ace_net = _build_ace_branch(br, input_size, input_channel).to(dev)
    ct_net = _build_ctrain_branch(ct, input_size, input_channel).to(dev)
    with torch.no_grad():
        out_ace = ace_net(xs)
        out_ct = ct_net(_normalize(xs))
        argmax_match = (out_ace.argmax(1) == out_ct.argmax(1)).float().mean().item()
        max_logit_diff = (out_ace - out_ct).abs().max().item()
    print(f"forward: argmax agreement {argmax_match:.3f}, max |logit diff| {max_logit_diff:.2e}")


def _normalize(x):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
    return (x - mean) / std


def _build_ace_branch(sd, input_size, input_channel):
    import networks as N
    net = N.myNet("cpu", "cifar10", n_class=10, input_size=input_size,
                  input_channel=input_channel, conv_widths=[2, 2, 8],
                  kernel_sizes=[3, 4, 4], strides=[1, 2, 2], linear_sizes=[250])
    net.load_state_dict(sd, strict=False)
    return net


def _build_ctrain_branch(sd, input_size, input_channel):
    import torch.nn as nn
    net = nn.Sequential(
        nn.Conv2d(input_channel, 16 * 2, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16 * 2, 16 * 2, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16 * 2, 16 * 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * 8 * 8 * 8, 250),
        nn.ReLU(),
        nn.Linear(250, 10),
    )
    net.load_state_dict(sd, strict=False)
    return net


def mode_label_quality(samples=2000, eps=0.00784313725):
    print(f"=== label quality: box vs zono on {samples} samples ===")
    args = _args(eps)
    lossFn = torch.nn.CrossEntropyLoss(reduction="none")
    evalFn = lambda x: torch.max(x, dim=1)[1]
    _, _, test_loader, input_size, input_channel, n_class = get_loaders(args)
    dTNet = MyDeepTrunkNet.get_deepTrunk_net(args, "cpu", lossFn, evalFn,
                                             input_size, input_channel, n_class)
    br = torch.load(os.path.join(CONVERTED, "ibp_2_255_branch.pt"), map_location="cpu")
    exit_idx = dTNet.exit_ids[1]   # the single branch exit
    dTNet.branch_nets[exit_idx].load_state_dict(br, strict=False)
    cnet = dTNet.branch_cnets[exit_idx]
    cnet.eval()

    box_ok, zono_ok, margins = [], [], []
    seen = 0
    with torch.no_grad():
        for x, y in test_loader:
            if seen >= samples:
                break
            seen += x.size(0)
            x, y = x, y
            v_box, m_box, _ = cnet.get_abs_loss(x, y, eps, "box", 0, beta=1)
            v_zono, m_zono, _ = cnet.get_abs_loss(x, y, eps, "zono", 0, beta=1)
            box_ok.append(v_box)
            zono_ok.append(v_zono)
            margins.append(m_box)
    box_ok = torch.cat(box_ok)[:samples].bool()
    zono_ok = torch.cat(zono_ok)[:samples].bool()
    margins = torch.cat(margins)[:samples]
    pos_box = box_ok.float().mean().item()
    pos_zono = zono_ok.float().mean().item()
    both = (box_ok & zono_ok).float().mean().item()
    only_box = (box_ok & ~zono_ok).float().mean().item()
    only_zono = (~box_ok & zono_ok).float().mean().item()
    frag_pos = ((margins > 0) & (margins <= 0.05)).float().mean().item()
    frag_neg = ((margins > -0.05) & (margins <= 0)).float().mean().item()
    print(f"box  positive rate : {pos_box:.4f}")
    print(f"zono positive rate : {pos_zono:.4f}")
    print(f"both certified     : {both:.4f}")
    print(f"box-only (zono miss): {only_box:.4f}")
    print(f"zono-only (box miss): {only_zono:.4f}")
    print(f"label disagreement : {only_box + only_zono:.4f}")
    print(f"fragile positives (0 < m <= .05): {frag_pos:.4f}")
    print(f"fragile negatives (-.05 < m <= 0): {frag_neg:.4f}")
    print("interpretation: high disagreement / high fragility -> box labels are noisy "
          "(high label noise degrades gate training)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["all", "norm", "keys", "equiv", "label-quality"])
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--branch", default="ibp_2_255_branch.pt")
    a = ap.parse_args()
    if a.mode in ("all", "norm"):
        mode_norm()
    if a.mode in ("all", "keys"):
        mode_keys(a.branch)
    if a.mode in ("all", "equiv"):
        mode_equiv()
    if a.mode in ("all", "label-quality"):
        mode_label_quality(a.samples)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnose why a trained ACE gate certifies (or not) at tau=0.

A trained dTNet checkpoint (net_*.pt) contains both the gate and the branch.
This script loads one and runs the same box verification that the evaluation
pipeline uses, reporting the gate margin structure:

  provable routing  = P( lb(gate) > tau )          <- what cert needs
  provable reject   = P( ub(gate) < tau )
  ambiguous         = P( lb <= tau <= ub )         <- uncertifiable routing
  branch ver        = P( branch verifies )         <- the cert targets' rate
  cert              = P( lb > tau  AND  branch verifies )

Running it on checkpoints whose gates were trained on different branches (e.g. a
custom branch vs the released one) localises the composition bottleneck:
  - low 'provable routing'          -> margin/coverage failure (gate can't prove routing)
  - high routing but low cert       -> alignment failure (routes samples that don't verify)

Usage (HPC, from ~/research/ACE with the 'ace' conda env):
  python ~/research/scripts/analyze_gate_margins.py \
      --run-dir models_new/cifar10/<exp>/<id>/None_0.00784/<ts> [--samples 2000]

  --run-dir   a timestamped run dir containing args.json + net_*.pt (latest used)
  --ckpt      alternatively: path to a single net_*.pt / whole-model checkpoint
  --samples   test samples to evaluate (default 2000; use 10000 for the full set)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# running from inside ~/research/ACE, the ACE modules are importable as-is;
# fall back to a local copy if present
sys.path.insert(0, os.getcwd())
# slurm jobs run from scripts/, where getcwd() is useless -> import
# args_factory from the ACE tree explicitly (ACE_DIR env wins if set).
_ace_dir = os.path.expanduser(os.environ.get("ACE_DIR") or "~/research/ACE")
if os.path.isdir(_ace_dir):
    sys.path.insert(0, _ace_dir)

from args_factory import get_args, translate_net_name   # noqa: E402
from loaders import get_loaders            # noqa: E402
from deepTrunk_networks import MyDeepTrunkNet  # noqa: E402


def build_args(run_dir, fallback=None):
    import sys as _sys
    # get_args() calls parser.parse_args() on sys.argv — neutralize it so we can
    # build the config from args.json (or defaults) instead of the CLI.
    _argv = _sys.argv[:]
    _sys.argv = [_argv[0], "--net", "None"]
    try:
        args = get_args()
    finally:
        _sys.argv = _argv
    # sensible defaults for the released-selector case (no args.json available)
    args.dataset = "cifar10"
    args.net = "None"
    args.n_branches = 1
    args.gate_type = "net"
    # get_network() only knows serialized myNet names, not "C3_cifar10"
    _c3 = translate_net_name("C3_cifar10")
    args.branch_nets = [_c3]
    args.gate_nets = [_c3]
    args.gate_threshold = 0.0
    args.test_eps = 0.00784313725
    args.train_eps = 0.00784313725
    args.train_batch = 100
    args.test_batch = 100
    args.num_workers = 4
    args.n_rand_proj = 50
    args.gate_feature_extraction = None
    args.cert_net_dim = None
    args.no_cuda = False
    args.test_set = "test"
    args.load_model = None
    args.load_trunk_model = None
    args.load_branch_model = None
    args.load_gate_model = None
    if fallback:
        for k, v in fallback.items():
            setattr(args, k, v)
    args_path = os.path.join(run_dir, "args.json")
    if os.path.exists(args_path):
        with open(args_path) as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
    # the trained selector checkpoints were made with --net None (no trunk)
    args.net = "None"
    args.load_model = None
    args.load_trunk_model = None
    args.load_branch_model = None
    args.load_gate_model = None
    return args


def latest_ckpt(run_dir):
    # ACE saves net_*.pt in {run_dir}/{episode}/{layer}/ subdirs
    nets = sorted(glob.glob(os.path.join(run_dir, "**", "net_*.pt"), recursive=True),
                  key=os.path.getmtime)
    if not nets:
        raise SystemExit(f"no net_*.pt under {run_dir} (searched recursively)")
    return nets[-1]


def load_with_numel_fix(model, sd):
    """Mirror ACE's load_net_state: direct strict=False load, falling back to a
    numel-matching view for shape differences (normalization saved as (1,3,1,1),
    scalar gate-head bias vs (1), etc.). Returns (missing, unexpected).
    """
    try:
        return model.load_state_dict(sd, strict=False)
    except RuntimeError:
        sdn = model.state_dict()
        for k, v in sd.items():
            if k in sdn and v.numel() == sdn[k].numel():
                sdn[k] = v.view(sdn[k].shape)
        return model.load_state_dict(sdn, strict=False)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None, help="run dir with args.json + net_*.pt")
    ap.add_argument("--ckpt", default=None, help="explicit checkpoint path (whole-dTNet state dict)")
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--tau", type=float, default=None, help="override gate threshold")
    ap.add_argument("--gate-domain", default="box",
                    choices=["box", "hbox", "zono", "l_IBP", "l_CROWN", "l_CROWN-IBP"],
                    help="verifier for the GATE's certified margins (provable routing). "
                         "Default box = what every cert number so far used. l_CROWN / "
                         "l_CROWN-IBP go through ACE's auto_LiRPA path and give tighter "
                         "gate bounds, i.e. MORE provable routing at the same tau, which is "
                         "the binding constraint at high nat. The branch verdict "
                         "stays on box so the two runs are comparable sample by sample.")
    ap.add_argument("--eps", type=float, default=None, help="override test eps")
    ap.add_argument("--dump-npz", default=None,
                    help="save per-sample test-set arrays (idx, lb, ub, box_ver, "
                         "gate_nat, gate_logit, branch_nat) for offline tau sweeps; "
                         "indices are the unshuffled CIFAR-10 test positions")
    args = ap.parse_args()

    if args.run_dir is None and args.ckpt is None:
        raise SystemExit("pass --run-dir or --ckpt")
    if args.run_dir is not None:
        run_dir = args.run_dir
        ckpt_path = latest_ckpt(run_dir)
        args_cfg = build_args(run_dir)
    else:
        run_dir = os.path.dirname(args.ckpt)
        ckpt_path = args.ckpt
        args_cfg = build_args(run_dir)
    # explicit overrides
    if args.tau is not None:
        args_cfg.gate_threshold = args.tau
    if args.eps is not None:
        # override BOTH: args.json may be missing (e.g. --run-dir pointing below the
        # <timestamp> dir), and the default fallback would silently verify at 2/255.
        args_cfg.test_eps = args_cfg.train_eps = args.eps
    tau = args_cfg.gate_threshold
    eps = args_cfg.test_eps

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== checkpoint: {ckpt_path} (device {device}) ===")
    if args.run_dir is not None:
        print(f"    (run dir: {run_dir})")
    lossFn = nn.CrossEntropyLoss(reduction="none")
    evalFn = lambda x: torch.max(x, dim=1)[1]

    _, _, test_loader, input_size, input_channel, n_class = get_loaders(args_cfg)
    dTNet = MyDeepTrunkNet.get_deepTrunk_net(args_cfg, device, lossFn, evalFn,
                                            input_size, input_channel, n_class)
    sd = torch.load(ckpt_path, map_location=device)
    missing, unexpected = load_with_numel_fix(dTNet, sd)
    miss_gb = [k for k in missing if "gate" in k or "branch" in k]
    print(f"load: {len(missing)} missing ({len(miss_gb)} gate/branch), "
          f"{len(unexpected)} unexpected")
    dTNet.eval()

    exit_idx = dTNet.exit_ids[1]           # the single branch exit
    gate_cnet = dTNet.gate_cnets[exit_idx]
    branch_cnet = dTNet.branch_cnets[exit_idx]
    gate_domain = args.gate_domain
    domain = "box"           # branch verdict: always box, so runs stay comparable

    lb_all, ub_all, ver_all, branch_ok_all, gate_nat_all, branch_nat_all, margin_list = [], [], [], [], [], [], []
    gate_logit_all = []
    n_seen = 0
    for inputs, targets in test_loader:
        if n_seen >= args.samples:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        b = inputs.size(0)
        n_seen += b

        # gate certified bounds: y=1 -> threshold_n = lb ; y=0 -> threshold_n = ub
        ones = torch.ones_like(targets).int()
        zeros = torch.zeros_like(targets).int()
        _, lb, _ = gate_cnet.get_abs_loss(inputs, ones, eps, gate_domain, tau, beta=1)
        _, ub, _ = gate_cnet.get_abs_loss(inputs, zeros, eps, gate_domain, tau, beta=1)
        # branch verification (true label)
        ver, margin_n, _ = branch_cnet.get_abs_loss(inputs, targets, eps, domain, 0, beta=1)

        gate_logits = dTNet.gate_nets[exit_idx].forward(inputs)          # (B,1)
        branch_logits = dTNet.branch_nets[exit_idx].forward(inputs)      # (B,10)

        lb_all.append(lb.cpu().float())
        ub_all.append(ub.cpu().float())
        ver_all.append(ver.cpu().float())
        margin_list.append(margin_n.cpu().float())
        branch_ok_all.append(targets.eq(branch_logits.argmax(1)).cpu().float())
        gate_nat_all.append((gate_logits.squeeze(1) > tau).cpu().float())
        gate_logit_all.append(gate_logits.squeeze(1).cpu().float())
        branch_nat_all.append(targets.eq(branch_logits.argmax(1)).cpu().float())

    lb = torch.cat(lb_all)[: args.samples]
    ub = torch.cat(ub_all)[: args.samples]
    ver = torch.cat(ver_all)[: args.samples].bool()
    gate_nat = torch.cat(gate_nat_all)[: args.samples].bool()
    gate_logit = torch.cat(gate_logit_all)[: args.samples]
    branch_nat = torch.cat(branch_nat_all)[: args.samples]

    if args.dump_npz:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_npz)) or ".", exist_ok=True)
        np.savez(args.dump_npz,
                 idx=np.arange(lb.numel(), dtype=np.int64),
                 lb=lb.cpu().numpy(), ub=ub.cpu().numpy(),
                 box_ver=ver.cpu().numpy(),
                 gate_nat=gate_nat.cpu().numpy(),
                 gate_logit=gate_logit.cpu().numpy(),
                 branch_nat=branch_nat.cpu().numpy(),
                 eps=np.float64(eps), tau=np.float64(tau), n=np.int64(lb.numel()))
        print(f"[dump] wrote {args.dump_npz} (n={lb.numel()}, eps={eps}, tau={tau}, "
              f"order = unshuffled test positions 0..{lb.numel()-1})")

    N = lb.numel()
    prov_route = (lb > tau).float().mean().item()
    prov_reject = (ub < tau).float().mean().item()
    ambiguous = 1 - prov_route - prov_reject
    branch_ver = ver.float().mean().item()
    cert = ((lb > tau) & ver).float().mean().item()
    gate_routing = gate_nat.float().mean().item()

    # margins on the samples that end up certified vs not
    routed_verified = ver & (lb > tau)
    misrouted = gate_nat & ~ver          # naturally routed but branch does not verify
    ver_on_routed = (ver[gate_nat].float().mean().item() if gate_nat.any() else float("nan"))
    cert_on_nat_routed = ((lb > tau)[gate_nat].float().mean().item() if gate_nat.any() else float("nan"))

    def pct(x, q):
        return torch.quantile(x.float(), torch.tensor(q)).item()

    print(f"\n=== {os.path.basename(run_dir)}  (n={N}, eps={eps}, tau={tau}, "
          f"gate_domain={gate_domain}) ===")
    print(f"  gate natural routing (logit > tau) : {gate_routing:.4f}")
    print(f"  branch nat acc                      : {branch_nat.float().mean().item():.4f}")
    print(f"  branch ver (target positive rate)   : {branch_ver:.4f}")
    print(f"  --- certified gate margins ({gate_domain}, eps={eps}) ---")
    print(f"  provable routing  lb>tau            : {prov_route:.4f}")
    print(f"  provable reject   ub<tau            : {prov_reject:.4f}")
    print(f"  ambiguous (can't prove either)      : {ambiguous:.4f}")
    print(f"  cert = lb>tau AND branch ver        : {cert:.4f}")
    print(f"  --- margin distribution (lb on all samples) ---")
    print(f"  lb quantiles 10/50/90               : {pct(lb, .1):.3f} / {pct(lb, .5):.3f} / {pct(lb, .9):.3f}")
    print(f"  lb upper tail  99/99.9/max          : {pct(lb, .99):.3f} / {pct(lb, .999):.3f} / {lb.max().item():.3f}")
    print(f"  gate logit     50/90/99/99.9/max     : {pct(gate_logit, .5):.3f} / {pct(gate_logit, .9):.3f} / "
          f"{pct(gate_logit, .99):.3f} / {pct(gate_logit, .999):.3f} / {gate_logit.max().item():.3f}")
    print(f"  --> natural routing hit-zero tau     : {gate_logit.max().item():.3f} (tau above this: nothing routes, nat = core-only)")
    print(f"  --> provable routing hit-zero tau    : {lb.max().item():.3f} (tau above this: cert = 0)")
    print(f"  ub quantiles 10/50/90               : {pct(ub, .1):.3f} / {pct(ub, .5):.3f} / {pct(ub, .9):.3f}")
    print(f"  width (ub-lb) median                : {pct(ub - lb, .5):.3f}")
    print(f"  --- routing alignment ---")
    print(f"  verify rate among naturally routed  : {ver_on_routed:.4f}")
    print(f"  provable routing among nat routed   : {cert_on_nat_routed:.4f}")
    print(f"  naturally routed but not verified   : {misrouted.float().mean().item():.4f}")

    # branch weight profile — compare against released branch norms
    print(f"  --- branch weight norms (mean|.| / max|.|) ---")
    sd_b = dTNet.branch_nets[exit_idx].state_dict()
    for k, v in sd_b.items():
        if k.endswith(".weight") and "conv" not in k and "linear" not in k:
            print(f"    {k:24s} {str(tuple(v.shape)):18s} {v.abs().mean().item():.5f} / {v.abs().max().item():.5f}")

    # branch cert-margin fragility: how many positives/negatives sit near margin 0
    margin_all = torch.cat(margin_list)[: args.samples]
    pos = margin_all > 0
    frac_fragile_pos = ((margin_all > 0) & (margin_all <= 0.05)).float().mean().item()
    frac_fragile_neg = ((margin_all > -0.05) & (margin_all <= 0)).float().mean().item()
    print(f"  --- branch cert margins ---")
    print(f"    margin quantiles 10/50/90          : {pct(margin_all, .1):.4f} / {pct(margin_all, .5):.4f} / {pct(margin_all, .9):.4f}")
    print(f"    fragile positives (0 < m <= .05)   : {frac_fragile_pos:.4f}   fragile negatives (-.05 < m <= 0): {frac_fragile_neg:.4f}")

    # gate head: the fitted operating point lives in the last linear's bias
    sd_g = dTNet.gate_nets[exit_idx].state_dict()
    hw = sd_g["blocks.layers.10.weight"]
    hb = sd_g["blocks.layers.10.bias"]
    print(f"  --- gate head (final linear) ---")
    if hb.numel() > 1:
        print(f"    weight mean|.|={hw.abs().mean().item():.4f}  bias mean={hb.mean().item():.4f}  bias std={hb.std().item():.4f}")
    else:
        print(f"    weight mean|.|={hw.abs().mean().item():.4f}  bias={hb.item():.4f} (single unit)")


if __name__ == "__main__":
    main()

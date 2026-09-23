#!/usr/bin/env python3
"""Standalone gate + branch verification (IBP / CROWN / alpha-CROWN), per test sample.

Why: ACE certifies gate and branch through its own relaxed wrappers, where auto_LiRPA's
alpha optimizer finds no parameters and silently skips (so `--cert-domain l_alpha` is
really plain CROWN on both). ACE's relaxed layers are bespoke, so this rebuilds the gate
and the branch as plain functional modules from the checkpoint weights, where IBP, CROWN
and alpha-CROWN all run. The branch margin uses ACE's C-matrix functional
(min_k lb[logit_y - logit_k], gen_labels.ace_c_spec) so the numbers sit next to ACE's box
cert.

Reports, per verifier:
    gate routing   = P( lb(gate logit) > tau )
    branch ver     = P( min_k lb[logit_y - logit_k] > 0 )
Dumps an npz with the per-sample arrays so combine_margins.py can compose them.

Usage (ace env, from ~/research/ACE):
  python gen_margins.py --run-dir <selector dir> --eps 0.00784313725 --tau 0.0 \
      --dump-npz results/dumps/margins_ibp_2_255.npz
"""
import argparse
import inspect
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ace = os.path.expanduser(os.environ.get("ACE_DIR") or "~/research/ACE")
if os.path.isdir(_ace):
    sys.path.insert(0, _ace)

from analyze_gate_margins import build_args, latest_ckpt, load_with_numel_fix  # noqa: E402
from gen_labels import ace_c_spec                                               # noqa: E402
from loaders import get_loaders                                                 # noqa: E402
from deepTrunk_networks import MyDeepTrunkNet                                   # noqa: E402
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm         # noqa: E402


def layer_state(net):
    """state_dict keys 'blocks.layers.<i>.*' -> '<i>.*'."""
    out = {}
    for k, v in net.state_dict().items():
        i = k.find("layers.")
        out[k[i + len("layers."):] if i >= 0 else k] = v
    return out


class PlainC3(nn.Module):
    """ACE C3 net (Normalization + conv/ReLU x3 + FC head) as plain functional ops, so
    auto_LiRPA recognises the ReLUs. C3_cifar10: conv 32/32/128, k 3/4/4, s 1/2/2 p 1, FC ->250->n."""
    def __init__(self, sd):
        super().__init__()
        self.register_buffer("mean", sd["0.mean"].detach().float().view(1, 3, 1, 1).clone())
        self.register_buffer("sigma", sd["0.sigma"].detach().float().view(1, 3, 1, 1).clone())
        for j, i in enumerate((1, 3, 5, 8, 10)):
            self.register_parameter(f"w{j}", nn.Parameter(sd[f"{i}.weight"].detach().float().clone(), requires_grad=False))
            self.register_parameter(f"b{j}", nn.Parameter(sd[f"{i}.bias"].detach().float().clone(), requires_grad=False))

    def forward(self, x):
        x = (x - self.mean) / self.sigma
        x = F.relu(F.conv2d(x, self.w0, self.b0, stride=1, padding=1))
        x = F.relu(F.conv2d(x, self.w1, self.b1, stride=2, padding=1))
        x = F.relu(F.conv2d(x, self.w2, self.b2, stride=2, padding=1))
        x = torch.flatten(x, 1)
        x = F.relu(F.linear(x, self.w3, self.b3))
        return F.linear(x, self.w4, self.b4)


def opt_kwargs(bm, steps):
    params = inspect.signature(bm.compute_bounds).parameters
    for name in ("iteration", "opt_steps"):
        if name in params:
            return {name: steps}
    return {}


def mins_flat(t):
    return (t.reshape(t.size(0), -1).min(dim=1).values if t.dim() > 1 else t.reshape(-1))


def _kw(okw):
    return {k: v for k, v in okw.items() if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--opt-steps", type=int, default=100)
    ap.add_argument("--dump-npz", default=None)
    a = ap.parse_args()

    if a.run_dir is None and a.ckpt is None:
        raise SystemExit("pass --run-dir or --ckpt")
    run_dir = a.run_dir or os.path.dirname(a.ckpt)
    ckpt = latest_ckpt(run_dir) if a.run_dir else a.ckpt
    cfg = build_args(run_dir)
    if a.eps is not None:
        cfg.test_eps = cfg.train_eps = a.eps
    eps = cfg.test_eps
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== margins: {ckpt} ===\n  eps={eps} tau={a.tau} device={device} opt_steps={a.opt_steps}")

    lossFn = nn.CrossEntropyLoss(reduction="none")
    evalFn = lambda x: torch.max(x, dim=1)[1]
    _, _, test_loader, input_size, input_channel, n_class = get_loaders(cfg)
    dTNet = MyDeepTrunkNet.get_deepTrunk_net(cfg, device, lossFn, evalFn, input_size, input_channel, n_class)
    load_with_numel_fix(dTNet, torch.load(ckpt, map_location=device))
    dTNet.eval()
    exit_idx = dTNet.exit_ids[1]

    gate = PlainC3(layer_state(dTNet.gate_nets[exit_idx])).to(device).eval()
    branch_sd = layer_state(dTNet.branch_nets[exit_idx])
    branch = PlainC3(branch_sd).to(device).eval()
    n_branch = branch_sd["10.weight"].shape[0]

    gbm = BoundedModule(gate, torch.rand(1, 3, 32, 32, device=device),
                        bound_opts={"optimize": True, "opt_steps": a.opt_steps}, device=device)
    bbm = BoundedModule(branch, torch.rand(1, 3, 32, 32, device=device),
                        bound_opts={"optimize": True, "opt_steps": a.opt_steps}, device=device)
    gkw = opt_kwargs(gbm, a.opt_steps)
    bkw = opt_kwargs(bbm, a.opt_steps)
    gkw["_steps"] = a.opt_steps
    print(f"  opt kwargs: gate={ {k:v for k,v in gkw.items() if not k.startswith('_')} } "
          f"branch={bkw} (n_branch={n_branch})")

    out = {k: [] for k in ("idx", "gate_logit", "gate_lb_box", "gate_lb_crown", "gate_lb_alpha",
                           "branch_ok", "branch_margin_box", "branch_margin_crown", "branch_margin_alpha")}
    n_seen = 0
    for inputs, targets in test_loader:
        if n_seen >= a.samples:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        b = inputs.size(0)
        out["idx"].append(torch.arange(n_seen, n_seen + b)); n_seen += b
        xb = BoundedTensor(inputs, PerturbationLpNorm(norm=float("inf"), eps=eps))
        with torch.no_grad():
            gb = gbm.compute_bounds(x=xb, IBP=True, method=None)[0]
            gc = gbm.compute_bounds(x=xb, IBP=False, method="backward")[0]
            ga = gbm.compute_bounds(x=xb, IBP=False, method="alpha-crown", **_kw(gkw))[0]
            out["gate_logit"].append(gate(inputs).reshape(-1).cpu())
            out["gate_lb_box"].append(mins_flat(gb).cpu())
            out["gate_lb_crown"].append(mins_flat(gc).cpu())
            out["gate_lb_alpha"].append(mins_flat(ga).cpu())
            c = ace_c_spec(targets, n_branch, device)
            for tag, call in (("box", dict(IBP=True, method=None)),
                              ("crown", dict(IBP=False, method="backward")),
                              ("alpha", dict(IBP=False, method="alpha-crown", **_kw(bkw)))):
                lb = bbm.compute_bounds(x=xb, C=c, **call)[0]
                out[f"branch_margin_{tag}"].append(mins_flat(lb).cpu())
            out["branch_ok"].append(targets.eq(branch(inputs).argmax(1)).cpu().float())

    arr = {k: torch.cat(v).numpy() for k, v in out.items()}
    n = len(arr["idx"])

    def grate(k):
        return float((arr[k] > a.tau).mean())

    def brate(k):
        return float((arr[k] > 0).mean())

    print(f"\n=== gate routing at tau={a.tau} (n={n}) ===")
    print(f"  natural          : {float((arr['gate_logit'] > a.tau).mean()):.4f}")
    for k in ("box", "crown", "alpha"):
        print(f"  {k:<6}           : {grate('gate_lb_' + k):.4f}")
    print(f"\n=== branch verification (n={n}) ===")
    print(f"  natural correct  : {float(arr['branch_ok'].mean()):.4f}")
    for k in ("box", "crown", "alpha"):
        print(f"  {k:<6}           : {brate('branch_margin_' + k):.4f}")

    if a.dump_npz:
        os.makedirs(os.path.dirname(os.path.abspath(a.dump_npz)) or ".", exist_ok=True)
        np.savez(a.dump_npz, eps=eps, tau=a.tau, **arr)
        print(f"[dump] wrote {a.dump_npz}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Standalone gate + branch verification (IBP / CROWN / alpha-CROWN), per test sample.

Why: ACE certifies gate and branch through its own relaxed wrappers, where auto_LiRPA's
alpha optimizer finds no parameters and silently skips (so `--cert-domain l_alpha` is
really plain CROWN on both). ACE's relaxed layers are bespoke, so this rebuilds the gate
and the branch as standard nn.Module layers from the checkpoint weights, where IBP, CROWN
and alpha-CROWN all run. The branch margin uses ACE's C-matrix functional
(min_k lb[logit_y - logit_k], ace_spec.ace_c_spec) so the numbers sit next to ACE's box
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
# NB: import ace_c_spec from ace_spec (pure torch), NOT from gen_labels. gen_labels
# imports train_branch -> CTRAIN -> abCROWN, whose bare `from utils import expand_path`
# collides with ACE's own utils.py that this script (via analyze_gate_margins) puts on
# the path -> ImportError. ace_spec has no ACE/CTRAIN deps.
from ace_spec import ace_c_spec                                                  # noqa: E402
from loaders import get_loaders                                                 # noqa: E402
from deepTrunk_networks import MyDeepTrunkNet                                   # noqa: E402
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm         # noqa: E402

GIB = 2 ** 30


def mem(tag, device):
    if device != "cuda":
        return
    print(f"    [mem {tag}] alloc={torch.cuda.memory_allocated()/GIB:.2f} "
          f"peak={torch.cuda.max_memory_allocated()/GIB:.2f} "
          f"reserved={torch.cuda.memory_reserved()/GIB:.2f} GiB", flush=True)


def layer_state(net):
    """state_dict keys 'blocks.layers.<i>.*' -> '<i>.*'."""
    out = {}
    for k, v in net.state_dict().items():
        i = k.find("layers.")
        out[k[i + len("layers."):] if i >= 0 else k] = v
    return out


class PlainC3(nn.Module):
    """ACE C3 net (Normalization + conv/ReLU x3 + FC head) as standard nn.Module layers.

    C3_cifar10: conv 32/32/128, k 3/4/4, s 1/2/2 p 1, FC ->250->n.

    NOTE: building this from *functional* ops (F.conv2d/F.linear) over frozen
    (requires_grad=False) parameters makes auto_LiRPA report "No optimizable
    parameters found. Will skip optimization.", and the `alpha` column then falls
    back to no better than `box` (a monotonicity violation: alpha must be >= box). Standard
    nn.Conv2d/nn.Linear layers with trainable parameters (the same shape gen_labels.py
    uses for CTRAIN branches, where alpha-CROWN optimises correctly) restore the alpha
    parameters. The weights are copied from ACE's checkpoint; alpha-CROWN optimises its
    own relaxation parameters, so these stay fixed.
    """
    def __init__(self, sd):
        super().__init__()
        self.register_buffer("mean", sd["0.mean"].detach().float().view(1, 3, 1, 1).clone())
        self.register_buffer("sigma", sd["0.sigma"].detach().float().view(1, 3, 1, 1).clone())
        w1, w3, w5 = (sd[f"{i}.weight"] for i in (1, 3, 5))
        b1, b3, b5 = (sd[f"{i}.bias"] for i in (1, 3, 5))
        w8, b8 = sd["8.weight"], sd["8.bias"]
        w10, b10 = sd["10.weight"], sd["10.bias"]
        self.c0 = nn.Conv2d(w1.shape[1], w1.shape[0], 3, stride=1, padding=1)
        self.c1 = nn.Conv2d(w3.shape[1], w3.shape[0], 4, stride=2, padding=1)
        self.c2 = nn.Conv2d(w5.shape[1], w5.shape[0], 4, stride=2, padding=1)
        self.f0 = nn.Linear(w8.shape[1], w8.shape[0])
        self.f1 = nn.Linear(w10.shape[1], w10.shape[0])
        with torch.no_grad():
            for mod, w, b in ((self.c0, w1, b1), (self.c1, w3, b3), (self.c2, w5, b5),
                              (self.f0, w8, b8), (self.f1, w10, b10)):
                mod.weight.copy_(w.float())
                mod.bias.copy_(b.float())

    def forward(self, x):
        x = (x - self.mean) / self.sigma
        x = F.relu(self.c0(x))
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.f0(x))
        return self.f1(x)


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


def optact(bm):
    """(#optimizable activations, #enabled for optimization).

    The second number is what actually feeds the alpha optimizer. If it is 0, auto_LiRPA
    prints 'No optimizable parameters found. Will skip optimization.' and the alpha column
    is meaningless. Nodes only become 'used' after a forward/backward has run, so this is
    meaningful when printed just before the alpha call, not at construction time.
    """
    try:
        return (len(getattr(bm, "optimizable_activations", [])),
                len(bm.get_enabled_opt_act()))
    except Exception as e:  # pragma: no cover
        return "ERR", str(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--opt-steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16,
                    help="verifier batch size. ACE's default test_loader uses 100, which "
                         "OOMs the alpha-CROWN backward by accumulation (gen_labels.py "
                         "docstring); 16 is the measured-safe value on the H100.")
    ap.add_argument("--mem-limit-gib", type=float, default=60.0,
                    help="abort cleanly (exit 2, nothing saved) if reserved memory exceeds "
                         "this, so a shard fails detectably instead of OOM-killing.")
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
    # Shrink the loader batch. build_args pins cfg.test_batch=100; the alpha-CROWN
    # backward scales with batch and accumulates, so 100 OOMs (~80 GiB) within a
    # batch even on a 93 GiB card. Same lesson as gen_labels.py: keep batches small.
    cfg.test_batch = a.batch_size
    _, _, test_loader, input_size, input_channel, n_class = get_loaders(cfg)
    print(f"  verifier batch-size={a.batch_size} mem-limit={a.mem_limit_gib} GiB")
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
    print(f"  [diag] built: gate optimizable/enabled={optact(gbm)} "
          f"branch optimizable/enabled={optact(bbm)}  (0 enabled at build time is normal)")

    out = {k: [] for k in ("idx", "gate_logit", "gate_lb_box", "gate_lb_crown", "gate_lb_alpha",
                           "branch_ok", "branch_margin_box", "branch_margin_crown", "branch_margin_alpha")}
    n_seen = 0
    batch_i = 0
    for inputs, targets in test_loader:
        if n_seen >= a.samples:
            break
        if device == "cuda":
            res = torch.cuda.memory_reserved() / GIB
            if res > a.mem_limit_gib:
                print(f"ABORT: reserved {res:.2f} GiB > limit {a.mem_limit_gib} GiB at "
                      f"batch {batch_i}; nothing saved", flush=True)
                sys.exit(2)
        inputs, targets = inputs.to(device), targets.to(device)
        b = inputs.size(0)
        out["idx"].append(torch.arange(n_seen, n_seen + b)); n_seen += b
        xb = BoundedTensor(inputs, PerturbationLpNorm(norm=float("inf"), eps=eps))
        if n_seen - b == 0:
            print(f"  [diag b0] gate   optimizable/enabled={optact(gbm)} (before alpha)")
            print(f"  [diag b0] branch optimizable/enabled={optact(bbm)} (before alpha)")
        c = ace_c_spec(targets, n_branch, device)
        with torch.no_grad():
            gb = gbm.compute_bounds(x=xb, IBP=True, method=None)[0]
            gc = gbm.compute_bounds(x=xb, IBP=False, method="backward")[0]
            bmb = bbm.compute_bounds(x=xb, C=c, IBP=True, method=None)[0]
            bmc = bbm.compute_bounds(x=xb, C=c, IBP=False, method="backward")[0]
            out["gate_logit"].append(gate(inputs).reshape(-1).cpu())
            out["gate_lb_box"].append(mins_flat(gb).cpu())
            out["gate_lb_crown"].append(mins_flat(gc).cpu())
            out["branch_margin_box"].append(mins_flat(bmb).cpu())
            out["branch_margin_crown"].append(mins_flat(bmc).cpu())
            out["branch_ok"].append(targets.eq(branch(inputs).argmax(1)).cpu().float())
        # alpha-CROWN MUST run with grad ENABLED: this is where the alpha variables are
        # optimised. Wrapping these calls in torch.no_grad() gives the bound tensors no
        # autograd path, so auto_LiRPA's optimizer finds no optimizable parameters and
        # silently skips ("No optimizable parameters found. Will skip optimization."),
        # leaving the alpha column identical to plain CROWN. gen_labels.py never wraps its
        # alpha call, which is why it optimises. Do NOT move these inside a no_grad block.
        ga = gbm.compute_bounds(x=xb, IBP=False, method="alpha-crown", **_kw(gkw))[0]
        out["gate_lb_alpha"].append(mins_flat(ga).detach().cpu())
        ba = bbm.compute_bounds(x=xb, C=c, IBP=False, method="alpha-crown", **_kw(bkw))[0]
        out["branch_margin_alpha"].append(mins_flat(ba).detach().cpu())

        # Free the per-batch graph/activations before the next batch. Without this the
        # alpha-CROWN intermediate bounds accumulate and the run OOMs (the traceback
        # lands in operators/clampmult.py, inside the optimizer's backward).
        if batch_i == 0 or (batch_i + 1) % 10 == 0:
            mem(f"b{batch_i+1} pre-clean", device)
        del xb, c, gb, gc, bmb, bmc, ga, ba, inputs, targets
        if device == "cuda":
            torch.cuda.empty_cache()
        if batch_i == 0:
            mem(f"b{batch_i+1} post-clean", device)
        batch_i += 1

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

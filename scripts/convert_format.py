#!/usr/bin/env python3
"""Convert CTRAIN-trained branch checkpoints into ACE myNet format.

Our CTRAIN branches are plain nn.Sequential state dicts; this script swaps the
branch weights into an ACE template checkpoint
(C3_ACE_Net_IBP_cert_cifar10_{2,8}_255.pt).

TWO output files per run — ACE has two DIFFERENT load paths and they need
different key layouts (getting this wrong loads NOTHING, silently, because
utils.load_net_state uses strict=False):

  <method>_<eps>.pt         whole-dTNet state dict, keys `branchNet_0.*` /
                            `gateNet_0.*`  -> use with `--load-model`
                            (deepTrunk_main eval / --train-mode cert)
  <method>_<eps>_branch.pt  BARE net state dict, keys `blocks.layers.*`
                            -> use with `--load-branch-model` and
                            `--load-gate-model` (selector training), same form
                            as released ./trained_models/C3_cifar10_IBP_2_255.pt

Outputs go to ~/research/converted/ (not into ACE's trained_models/, which
holds the released checkpoints).

Usage:
    python convert_format.py [--method sabr --eps 0.00784313725]   # one run
    python convert_format.py --all                                  # whole matrix
"""
import argparse
import os
import torch

ACE_DIR = os.path.expanduser("~/research/ACE/trained_models")
RUNS = os.path.expanduser("~/research/runs")
OUT = os.path.expanduser("~/research/converted")

TEMPLATES = {
    "2_255": os.path.join(ACE_DIR, "C3_ACE_Net_IBP_cert_cifar10_2_255.pt"),
    "8_255": os.path.join(ACE_DIR, "C3_ACE_Net_IBP_cert_cifar10_8_255.pt"),
}
# (method, train-eps RAW, ace eps suffix)
MATRIX = [
    ("sabr",    0.00784313725, "2_255"),
    ("sabr",    0.03137254901, "8_255"),
    ("mtl_ibp", 0.00784313725, "2_255"),
    ("mtl_ibp", 0.03137254901, "8_255"),
    ("ibp",     0.00784313725, "2_255"),
    ("ibp",     0.03137254901, "8_255"),
    ("crown_ibp", 0.00784313725, "2_255"),
    ("crown_ibp", 0.03137254901, "8_255"),
]

# CTRAIN nn.Sequential layer indices -> ACE myNet block layer indices
# (ACE template has a Normalization block at layers.0; our Sequential starts at 0)
SRC_LAYERS = [0, 2, 4, 7, 9]
CONV, LINEAR = {0, 2, 4}, {7, 9}


def swap_branch(tpl, sd, branch):
    """Copy CTRAIN branch weights into the ACE template under `branch`."""
    n = 0
    for i in SRC_LAYERS:
        d = i + 1
        w, b = sd[f"{i}.weight"], sd[f"{i}.bias"]
        sub = "conv" if i in CONV else "linear"
        for dst in (f"{branch}.blocks.layers.{d}.weight", f"{branch}.blocks.layers.{d}.{sub}.weight"):
            assert tuple(tpl[dst].shape) == tuple(w.shape), f"shape {dst}"
            tpl[dst] = w; n += 1
        for dst in (f"{branch}.blocks.layers.{d}.bias", f"{branch}.blocks.layers.{d}.{sub}.bias",
                    f"{branch}.blocks.layers.{d}.b"):
            assert tuple(tpl[dst].shape) == tuple(b.shape), f"shape {dst}"
            tpl[dst] = b; n += 1
    return n


def convert_one(method, eps, eps_sfx, runs_dir=None, out_branch=None):
    branch = "branchNet_0"
    tpl = torch.load(TEMPLATES[eps_sfx], map_location="cpu")
    runs_dir = runs_dir or os.path.join(RUNS, f"{method}_eps{eps:.8f}")
    p = os.path.join(runs_dir, "final_model_state.pt")
    assert os.path.exists(p), f"missing plain model: {p}"
    n = swap_branch(tpl, torch.load(p, map_location="cpu"), branch)

    if out_branch is None:
        os.makedirs(OUT, exist_ok=True)
        out_branch = os.path.join(OUT, f"{method}_{eps_sfx}_branch.pt")
    os.makedirs(os.path.dirname(os.path.abspath(out_branch)), exist_ok=True)
    out = out_branch[:-len("_branch.pt")] + ".pt" if out_branch.endswith("_branch.pt") else out_branch + ".pt"
    torch.save(tpl, out)

    # bare branch state dict for --load-branch-model / --load-gate-model
    prefix = branch + "."
    bare = {k[len(prefix):]: v for k, v in tpl.items() if k.startswith(prefix)}
    assert bare, f"no {prefix}* keys in template"
    torch.save(bare, out_branch)

    # sanity: first conv layer of branch matches source
    ok = torch.equal(tpl[f"{branch}.blocks.layers.1.weight"], torch.load(p, map_location="cpu")["0.weight"])
    print(f"OK {method} {eps_sfx}: {n} tensors swapped -> {os.path.basename(out)} "
          f"+ {os.path.basename(out_branch)} ({len(bare)} bare keys) | verified: {ok}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices={m for m, _, _ in MATRIX})
    ap.add_argument("--eps", type=float, help="train eps (raw)")
    ap.add_argument("--all", action="store_true", help="convert whole matrix")
    ap.add_argument("--runs-dir", default=None,
                    help="dir with final_model_state.pt (default: runs/<method>_eps<eps>.8f)")
    ap.add_argument("--output", default=None,
                    help="bare branch output path (default: converted/<method>_<eps>_branch.pt)")
    args = ap.parse_args()

    if args.all:
        for row in MATRIX:
            convert_one(*row)
    else:
        assert args.method and args.eps is not None, "pass --method + --eps, or --all"
        matches = [r for r in MATRIX if r[0] == args.method and r[1] == args.eps]
        assert matches, f"no matrix row for {args.method} {args.eps}"
        convert_one(*matches[0], runs_dir=args.runs_dir, out_branch=args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate alpha-CROWN labels for a CTRAIN branch (shardable, memory-guarded).

Features:
  * `--start` / `--samples` slice the training set, so a job array can shard one branch
    across tasks. Each task writes its own CSV with GLOBAL sample indices, merged later.
  * Memory diagnostics: reports alloc/peak/reserved BEFORE and AFTER per-batch cleanup,
    so a real leak can be told apart from an allocator cache.
  * `--mem-limit-gib` aborts cleanly (exit 2, nothing saved) instead of OOM-killing the
    task, so a shard fails detectably rather than corrupting the run.
  * `--multi-method` additionally reports the margin under a cheap method first, for the
    label-flip table (box/IBP vs CROWN vs alpha-CROWN on the same samples).

alpha-CROWN throughput: batch 4 = 1.06 s/sample, batch 16 = 0.53
s/sample, identical labels (4/8) at both batch sizes. batch 100 on the plain-CROWN path
OOMs by accumulation. Keep batches small, shard long runs.

Usage (ace env, GPU):
  python gen_labels.py \
    --ckpt ~/research/runs/ibp_eps0.00784314/final_model_state.pt \
    --eps 0.00784313725 --train --start 0 --samples 12500 \
    --batch-size 16 --optimize --opt-steps 100 \
    --output ~/research/labels/labels_alpha_crown_ibp_2_255_shard0of4.csv
"""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import inspect
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

AUTOLIRPA = os.path.expanduser("~/research/auto_LiRPA_Repo")
if os.path.isdir(AUTOLIRPA):
    sys.path.insert(0, AUTOLIRPA)
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm  # noqa: E402

from train_branch import build_model, IN_SHAPE  # noqa: E402
from ace_spec import ace_c_spec  # noqa: E402

# NOTE: keep the leading 1 (shape (1,3,1,1)). This auto_LiRPA version's backward pass
# asserts const.ndim == 4 in operators/bivariate.py:_multiply_by_const. The "Constant
# operand has batch dimension" warning for the normalization node is benign.
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)

GIB = 2 ** 30


class NormalizedBranch(nn.Module):
    def __init__(self, branch):
        super().__init__()
        self.branch = branch
        self.register_buffer("mean", MEAN)
        self.register_buffer("std", STD)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.branch(x)


def load_ctrain_branch(ckpt_path, device, arch="ace_c3"):
    model = build_model(arch, IN_SHAPE).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt)
    return model


def mem(tag, device):
    if device != "cuda":
        return
    print(f"    [mem {tag}] alloc={torch.cuda.memory_allocated()/GIB:.2f} "
          f"peak={torch.cuda.max_memory_allocated()/GIB:.2f} "
          f"reserved={torch.cuda.memory_reserved()/GIB:.2f} GiB", flush=True)


def resolve_call(model, want_opt, opt_steps, method):
    """compute_bounds kwargs this installed auto_LiRPA actually accepts."""
    params = inspect.signature(model.compute_bounds).parameters
    kw = {"method": method}
    used = None
    if want_opt:
        for name in ("iteration", "opt_steps"):
            if name in params:
                kw[name] = opt_steps
                used = name
                break
    return kw, used, list(params.keys())


# ace_c_spec lives in ace_spec.py (pure torch, no CTRAIN) so that callers which only
# need the C-matrix do not pull in train_branch -> CTRAIN -> abCROWN. See ace_spec.py.


def margins(model, x_b, y, call_kw, ace_spec=False, n_class=10):
    """Return the certified margin per sample; >0 means certified.

    ace_spec=False (default, what the training labels used):
        lb[y] - max_{k!=y} ub[k], from the bounds on the individual logits. Sound, but a
        *different and stricter* functional than ACE's box verifier uses, so its rate is
        NOT comparable to ACE's cert (measured: 0.341 vs ACE box 0.486 on IBP
        2/255, with alpha_only ~ 0 and box_only 0.147).
    ace_spec=True:
        min_k lb[logit_y - logit_k] with ACE's C-matrix, i.e. exactly what
        `networks.py:verify(domain='box')` computes (threshold_n = min over I of
        concretize()[0]). This is the functional that makes alpha-CROWN >= box
        monotone, so it is the column to quote next to ACE's cert.
    """
    if ace_spec:
        c = ace_c_spec(y, n_class, x_b.device)
        lb, ub = model.compute_bounds(x=x_b, C=c, **call_kw)
        return lb.view(y.size(0), -1).min(dim=1).values.detach(), lb, ub
    lb, ub = model.compute_bounds(x=x_b, **call_kw)
    lb_y = lb.gather(1, y.view(-1, 1)).squeeze(1)
    ub_others = ub.clone()
    ub_others.scatter_(1, y.view(-1, 1), float("-inf"))
    return (lb_y - ub_others.max(1).values).detach(), lb, ub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default="ace_c3")
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--samples", type=int, default=50000)
    ap.add_argument("--start", type=int, default=0,
                    help="first sample index in the chosen split (for sharding)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--opt-steps", type=int, default=100)
    ap.add_argument("--alpha-method", default="alpha-crown")
    ap.add_argument("--prefilter-method", default=None,
                    help="optional cheap method (e.g. crown) evaluated first on the same "
                         "samples, to record the label-flip table; its columns are kept")
    ap.add_argument("--cheap-only", action="store_true",
                    help="run ONLY --prefilter-method and use its verdict as the label "
                         "(fast; for the box/CROWN side of the flip table)")
    ap.add_argument("--ace-spec", action="store_true",
                    help="use ACE's C-matrix functional min_k lb[logit_y - logit_k] "
                         "(networks.py:93) instead of lb[y] - max_k ub[k]. Use this for any "
                         "column that will be compared with ACE's cert: it is the same "
                         "functional, so alpha-CROWN >= box is guaranteed.")
    ap.add_argument("--n-class", type=int, default=10)
    ap.add_argument("--train", action="store_true", help="use training split (default: test)")
    ap.add_argument("--mem-limit-gib", type=float, default=60.0,
                    help="abort cleanly (exit 2) if reserved memory exceeds this")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"ckpt: {args.ckpt}")
    print(f"eps: {args.eps}, device: {device}")
    print(f"optimize: {args.optimize}, opt_steps: {args.opt_steps}, method: {args.alpha_method}")
    print(f"prefilter_method: {args.prefilter_method}")
    print(f"margin functional: {'ACE C-matrix (min_k lb[logit_y - logit_k])' if args.ace_spec else 'lb[y] - max_k ub[k] (strict)'}")

    branch = load_ctrain_branch(args.ckpt, device, args.arch).eval()
    model = BoundedModule(NormalizedBranch(branch).to(device),
                          torch.randn(1, 3, 32, 32).to(device),
                          bound_opts=({"optimize": True, "opt_steps": args.opt_steps}
                                      if args.optimize else {}))
    call_kw, used, sig = resolve_call(model, args.optimize, args.opt_steps, args.alpha_method)
    print(f"[api] signature params: {sig}")
    print(f"[api] bound_opts: {'{optimize, opt_steps}' if args.optimize else '{}'}")
    print(f"[api] compute_bounds kwargs: {call_kw} (opt-step kwarg name: {used})")

    transform = transforms.Compose([transforms.ToTensor()])
    full = datasets.CIFAR10(root=os.path.expanduser("~/research/data"),
                            train=args.train, download=True, transform=transform)
    n_total = len(full)
    lo = max(0, args.start)
    hi = min(n_total, lo + args.samples) if args.samples > 0 else n_total
    dataset = torch.utils.data.Subset(full, list(range(lo, hi)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    print(f"split={'train' if args.train else 'test'} range=[{lo},{hi}) n={len(dataset)} "
          f"batch={args.batch_size}")

    labels, idxs = [], []
    pref_lb, pref_ub = [], []
    t0 = time.time()
    cursor = lo
    for batch_idx, (x, y) in enumerate(loader):
        res_gib = torch.cuda.memory_reserved() / GIB if device == "cuda" else 0.0
        if res_gib > args.mem_limit_gib:
            print(f"ABORT: reserved {res_gib:.2f} GiB > limit {args.mem_limit_gib} GiB at "
                  f"batch {batch_idx}; nothing saved", flush=True)
            sys.exit(2)
        x, y = x.to(device), y.to(device)
        x_b = BoundedTensor(x, PerturbationLpNorm(norm=np.inf, eps=args.eps))

        m_pre = None
        if args.prefilter_method:
            m_pre, _, _ = margins(model, x_b, y, {"method": args.prefilter_method},
                                  ace_spec=args.ace_spec, n_class=args.n_class)
            pref_lb.extend((m_pre > 0).tolist())

        if args.cheap_only:
            if m_pre is None:
                sys.exit("--cheap-only needs --prefilter-method")
            m = m_pre
        else:
            m, lb, ub = margins(model, x_b, y, call_kw,
                                ace_spec=args.ace_spec, n_class=args.n_class)
        labels.extend((m > 0).tolist())
        idxs.extend(range(cursor, cursor + x.size(0)))
        cursor += x.size(0)

        if batch_idx == 0 or (batch_idx + 1) % 10 == 0:
            done = len(labels)
            print(f"  [{batch_idx+1}/{len(loader)}] pos={sum(labels)}/{done} "
                  f"({sum(labels)/done:.4f}) {done/(time.time()-t0):.2f} samples/s", flush=True)
        mem(f"b{batch_idx+1} pre-clean", device)
        del m, x_b, x, y
        if device == "cuda":
            torch.cuda.empty_cache()
        if batch_idx == 0:
            mem(f"b{batch_idx+1} post-clean", device)

    elapsed = time.time() - t0
    n, npos = len(labels), sum(labels)
    if n == 0:
        sys.exit("no samples processed")
    print(f"certified: {npos}/{n} ({npos/n:.4f})  time {elapsed:.1f}s "
          f"({elapsed/n:.3f} s/sample)")

    cols = {"index": idxs, "label": labels}
    if args.prefilter_method:
        cols[f"label_{args.prefilter_method}"] = pref_lb
    df = pd.DataFrame(cols).sort_values("index")
    tmp = args.output + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(tmp, index=False)
    os.replace(tmp, args.output)
    print(f"SAVED_OK: {args.output} ({len(df)} rows, pos_rate={npos/n:.4f})")
    if args.prefilter_method and n:
        pf = sum(pref_lb)
        flips = sum(1 for a, b in zip(pref_lb, labels) if a != b)
        print(f"[flip] {args.prefilter_method}->alpha: positive {pf}/{n} ({pf/n:.4f}) "
              f"-> {npos}/{n} ({npos/n:.4f}); label flips {flips}/{n} ({flips/n:.4f})")


if __name__ == "__main__":
    main()

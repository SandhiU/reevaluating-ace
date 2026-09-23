#!/usr/bin/env python3
"""Train certified networks (SABR / MTL-IBP / IBP) on CIFAR-10 via CTRAIN.

ACE thesis certified training. Full CIFAR-10 (no subset). Per run it saves:
  - ckpt/model_*.pt        CTRAIN wrapper checkpoints (every --ckpt-interval epochs)
  - final_model_state.pt   raw model state_dict -> input for the ACE bridge
  - results.json           config + final std/cert/adv accuracy
  - logs/<run>.csv         ExperimentLogger CSV

Examples:
  python train_branch.py --method sabr      --eps 0.00784313725 --epochs 160   # SABR eps=2/255
  python train_branch.py --method mtl_ibp   --eps 0.03137254901 --epochs 160   # MTL-IBP eps=8/255
  python train_branch.py --method crown_ibp --eps 0.00784313725 --epochs 160   # CROWN-IBP eps=2/255

Architectures (--arch):
  - ace_c2 / ace_c3 / ace_c5 (DEFAULT: ace_c3): mirror ACE's branch nets exactly
    (networks.py myNet, translate_net_name for cifar: C3_cifar10 =
    myNet__2_2_8__3_4_4__1_2_2__1__0__250 -> convs 32/32/128, k 3/4/4, s 1/2/2,
    FC 250->10). No Normalization block inside (CTRAIN loaders pre-normalize;
    the ACE bridge drops ACE's built-in norm to avoid double-normalizing).
  - cnn7_shi: CTRAIN's standard 7-layer net (NOT bridgeable into ACE as-is —
    kept only as a playground option).

Notes:
  - CTRAIN wrapper names are the CURRENT installed ones: SABRModelWrapper,
    MTLIBPModelWrapper, ShiIBPModelWrapper, CrownIBPModelWrapper
    (checked 18 Aug 2026 on HPC).
  - --alpha maps to sabr_subselection_ratio (SABR) / mtl_ibp_alpha (MTL-IBP);
    omitted -> the wrapper's paper default (0.2 / 0.5). IBP and CROWN-IBP
    have no alpha.
  - Output dirs follow the convention runs/<method>_eps<eps:.8f>/ and feed
    convert_format.py (matrix expects raw eps 0.00784313725 / 0.03137254901).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from CTRAIN.data_loaders import load_cifar10
from CTRAIN.model_definitions import CNN7_Shi
from CTRAIN.model_wrappers import (
    SABRModelWrapper,
    MTLIBPModelWrapper,
    ShiIBPModelWrapper,
    CrownIBPModelWrapper,
)
from helpers.logging import ExperimentLogger

EPS_2_255 = 0.00784313725
EPS_8_255 = 0.03137254901
IN_SHAPE = [3, 32, 32]

WRAPPERS = {
    "sabr": SABRModelWrapper,
    "mtl_ibp": MTLIBPModelWrapper,
    "ibp": ShiIBPModelWrapper,
    "crown_ibp": CrownIBPModelWrapper,
}
ALPHA_KW = {
    "sabr": "sabr_subselection_ratio",
    "mtl_ibp": "mtl_ibp_alpha",
    "ibp": None,
    "crown_ibp": None,
}

# ACE translate_net_name() outputs for cifar (networks.py myNet, scale_width x16).
ACE_CIFAR_ARCHS = {
    "ace_c2": dict(conv_widths=[2, 2], kernel_sizes=[4, 4], strides=[2, 2],
                   paddings=[1, 1], linear_sizes=[200]),
    "ace_c3": dict(conv_widths=[2, 2, 8], kernel_sizes=[3, 4, 4], strides=[1, 2, 2],
                   paddings=[1, 1, 1], linear_sizes=[250]),
    "ace_c5": dict(conv_widths=[4, 4, 8, 8, 8], kernel_sizes=[3, 3, 3, 3, 3],
                   strides=[1, 1, 2, 1, 1], paddings=[1, 1, 1, 1, 1], linear_sizes=[512]),
}


def build_ace_myNet(cfg, n_class=10, in_ch=3, in_size=32):
    """Mirror ACE myNet for cifar (width x16) WITHOUT the Normalization block."""
    layers, n_ch, n = [], in_ch, in_size
    for w, k, s, p in zip(cfg["conv_widths"], cfg["kernel_sizes"], cfg["strides"], cfg["paddings"]):
        layers += [nn.Conv2d(n_ch, w * 16, k, stride=s, padding=p), nn.ReLU()]
        n_ch = w * 16
        n = (n + 2 * p - (k - 1) - 1) // s + 1
    layers += [nn.Flatten(), nn.Linear(n_ch * n * n, cfg["linear_sizes"][0]), nn.ReLU(),
               nn.Linear(cfg["linear_sizes"][0], n_class)]
    return nn.Sequential(*layers)


def build_model(arch, in_shape):
    if arch == "cnn7_shi":
        return CNN7_Shi(in_shape=in_shape)
    if arch in ACE_CIFAR_ARCHS:
        return build_ace_myNet(ACE_CIFAR_ARCHS[arch])
    raise ValueError(f"unknown --arch {arch!r} (have: cnn7_shi, ace_c2, ace_c3, ace_c5)")


def _coerce(v):
    """Parse a --set value: none/()/comma-list -> tuple, int, float, else str."""
    s = v.strip()
    if s in ("none", "()"):
        return ()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
        return tuple(int(x) for x in s.split(",")) if s else ()
    if "," in s:
        return tuple(int(x) for x in s.split(","))
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=sorted(WRAPPERS), required=True)
    ap.add_argument("--eps", type=float, required=True,
                    help=f"2/255={EPS_2_255}, 8/255={EPS_8_255}")
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--alpha", type=float, default=None,
                    help="SABR: sabr_subselection_ratio; MTL-IBP: mtl_ibp_alpha (default = wrapper default)")
    ap.add_argument("--arch", default="ace_c3")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr-milestones", type=str, default="80,90",
                    help="comma-separated LR decay epochs (SABR paper: 120,140)")
    ap.add_argument("--entropy-loss-weight", type=float, default=0.0,
                    help="ACE-style joint entropy loss weight (0 = off; SABR/MTL-IBP only)")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--data-root", default=os.path.expanduser("~/research/data"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-samples", type=int, default=2000)
    ap.add_argument("--ckpt-interval", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="extra CTRAIN wrapper kwarg (repeatable), e.g. --set pgd_steps=8")
    args = ap.parse_args()

    if args.run_dir is None:
        args.run_dir = os.path.expanduser(f"~/research/runs/{args.method}_eps{args.eps:.8f}")
    os.makedirs(os.path.join(args.run_dir, "ckpt"), exist_ok=True)
    os.makedirs(os.path.join(args.run_dir, "logs"), exist_ok=True)
    os.makedirs(args.data_root, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log = ExperimentLogger(
        name=f"train_{args.method}",
        output_dir=os.path.join(args.run_dir, "logs"),
        extra_meta={
            "method": args.method, "eps": args.eps, "epochs": args.epochs,
            "arch": args.arch, "alpha": args.alpha, "seed": args.seed,
        },
    )

    print(f"[train_branch] loading CIFAR-10 (root={args.data_root}) ...", flush=True)
    train_loader, test_loader = load_cifar10(
        batch_size=args.batch_size, val_split=False, data_root=args.data_root,
    )

    model = build_model(args.arch, IN_SHAPE).to(args.device)
    print(f"[train_branch] {args.arch}: {sum(p.numel() for p in model.parameters()):,} params", flush=True)

    wrapper_kwargs = dict(
        model=model, input_shape=IN_SHAPE, eps=args.eps, num_epochs=args.epochs,
        lr=args.lr, device=args.device,
        checkpoint_save_path=os.path.join(args.run_dir, "ckpt", "model"),
        lr_decay_kwargs={"milestones": tuple(int(m) for m in args.lr_milestones.split(",")),
                         "gamma": 0.2},
        checkpoint_save_interval=args.ckpt_interval,
    )
    alpha_kw = ALPHA_KW[args.method]
    if alpha_kw is not None and args.alpha is not None:
        wrapper_kwargs[alpha_kw] = args.alpha

    # passthrough of the rest of the wrapper hyperparameters (see branch_hp in config.sh)
    for item in args.set:
        key, sep, val = item.partition("=")
        if not sep:
            raise SystemExit(f"--set needs KEY=VALUE, got {item!r}")
        parsed = _coerce(val)
        if key == "lr_decay_factor":
            wrapper_kwargs["lr_decay_kwargs"]["gamma"] = parsed
        else:
            wrapper_kwargs[key] = parsed

    if args.entropy_loss_weight:
        from entropy_trainers import enable_entropy_loss
        enable_entropy_loss(args.entropy_loss_weight)

    wrapper = WRAPPERS[args.method](**wrapper_kwargs)

    print(f"[train_branch] training {args.method} (eps={args.eps}, {args.epochs} epochs) ...", flush=True)
    t0 = time.time()
    wrapper.train_model(train_loader)
    train_seconds = time.time() - t0

    print(f"[train_branch] final evaluate ({args.eval_samples} test samples) ...", flush=True)
    std_acc, cert_acc, adv_acc = wrapper.evaluate(test_loader, test_samples=args.eval_samples)

    model_path = os.path.join(args.run_dir, "final_model_state.pt")
    torch.save(model.state_dict(), model_path)

    results = {
        "method": args.method, "eps": args.eps, "epochs": args.epochs,
        "arch": args.arch,
        "alpha": wrapper_kwargs.get(alpha_kw) if alpha_kw else None,
        "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
        "std_acc": float(std_acc), "cert_acc": float(cert_acc), "adv_acc": float(adv_acc),
        "eval_samples": args.eval_samples,
        "train_seconds": round(train_seconds, 1),
        "final_model": model_path,
    }
    with open(os.path.join(args.run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.log_final(results)

    print(f"[train_branch] DONE  std={std_acc:.4f} cert={cert_acc:.4f} adv={adv_acc:.4f}  ({train_seconds / 60:.1f} min)", flush=True)
    print(f"[train_branch] run dir: {args.run_dir}", flush=True)


if __name__ == "__main__":
    main()

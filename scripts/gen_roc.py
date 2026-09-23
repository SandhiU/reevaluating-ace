#!/usr/bin/env python3
"""
ROC curve of the SelNet gate head.

For each test sample the gate produces a continuous logit; the ground truth
is whether the branch box-certifies that sample at the given epsilon.
Sweeping the threshold tau yields the ROC curve (TPR vs FPR).

Usage (run from ~/research/ACE with ace env):
  python ../scripts/gen_roc.py --run-dir <dir with args.json + net_*.pt>
  python ../scripts/gen_roc.py --ckpt <net_*.pt> --eps 0.00784 --label ibp2
"""

import argparse, glob, json, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from args_factory import get_args, translate_net_name
from loaders import get_loaders
from deepTrunk_networks import MyDeepTrunkNet

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def build_args(run_dir):
    import sys as _sys
    _argv = _sys.argv[:]
    _sys.argv = [_argv[0], "--net", "None"]
    try:
        args = get_args()
    finally:
        _sys.argv = _argv
    args.dataset = "cifar10"
    args.net = "None"
    args.n_branches = 1
    args.gate_type = "net"
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
    args_path = os.path.join(run_dir, "args.json")
    if os.path.exists(args_path):
        with open(args_path) as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
    args.net = "None"
    args.load_model = None
    args.load_trunk_model = None
    args.load_branch_model = None
    args.load_gate_model = None
    return args


def latest_ckpt(run_dir):
    nets = sorted(glob.glob(os.path.join(run_dir, "**", "net_*.pt"), recursive=True),
                  key=os.path.getmtime)
    if not nets:
        raise SystemExit(f"no net_*.pt under {run_dir}")
    return nets[-1]


def load_with_numel_fix(model, sd):
    try:
        return model.load_state_dict(sd, strict=False)
    except RuntimeError:
        sdn = model.state_dict()
        for k, v in sd.items():
            if k in sdn and v.numel() == sdn[k].numel():
                sdn[k] = v.view(sdn[k].shape)
        return model.load_state_dict(sdn, strict=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--label", default=None)
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
    if args.eps is not None:
        args_cfg.test_eps = args.eps
    eps = args_cfg.test_eps
    tau = args_cfg.gate_threshold

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"checkpoint: {ckpt_path}")
    print(f"eps: {eps}, device: {device}")

    lossFn = nn.CrossEntropyLoss(reduction="none")
    evalFn = lambda x: torch.max(x, dim=1)[1]
    _, _, test_loader, input_size, input_channel, n_class = get_loaders(args_cfg)

    dTNet = MyDeepTrunkNet.get_deepTrunk_net(args_cfg, device, lossFn, evalFn,
                                              input_size, input_channel, n_class)
    sd = torch.load(ckpt_path, map_location=device)
    missing, unexpected = load_with_numel_fix(dTNet, sd)
    print(f"load: {len(missing)} missing, {len(unexpected)} unexpected")
    dTNet.eval()

    exit_idx = dTNet.exit_ids[1]
    gate_cnet = dTNet.gate_cnets[exit_idx]
    branch_cnet = dTNet.branch_cnets[exit_idx]
    domain = "box"

    # ---------- collect per-sample gate logit + branch cert ----------
    gate_logits_all = []
    cert_labels_all = []  # 1 = branch certifies, 0 = does not
    n_seen = 0
    N = min(args.samples, len(test_loader.dataset))

    with torch.no_grad():
        for inputs, targets in test_loader:
            if n_seen >= N:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            b = inputs.size(0)
            if n_seen + b > N:
                inputs, targets = inputs[:N - n_seen], targets[:N - n_seen]
                b = inputs.size(0)
            n_seen += b

            # raw gate logit (the score the gate outputs)
            gate_logit = dTNet.gate_nets[exit_idx].forward(inputs).squeeze(-1)  # (B,)

            # branch verification (true label): lb on classification margin
            ver, margin_n, _ = branch_cnet.get_abs_loss(inputs, targets, eps, domain, 0, beta=1)

            gate_logits_all.append(gate_logit.cpu().float())
            cert_labels_all.append((ver > 0).cpu().float())

    gate_logits = torch.cat(gate_logits_all)[:N].numpy()
    cert_labels = torch.cat(cert_labels_all)[:N].numpy()

    n_cert = int(cert_labels.sum())
    n_total = len(cert_labels)
    print(f"\n{n_total} samples, {n_cert} certified ({n_cert/n_total:.4f}), "
          f"{n_total - n_cert} not certified")

    # ---------- ROC ----------
    fpr, tpr, thresholds = roc_curve(cert_labels, gate_logits)
    auc = roc_auc_score(cert_labels, gate_logits)
    print(f"AUC = {auc:.4f}")

    # ---------- operating points ----------
    key_tau = [0, -0.5, -0.9, -1.0, -2.0, -5.0, -10.0, -50.0]
    print(f"\n{'tau':>8s}  {'TPR':>6s}  {'FPR':>6s}  {'route':>6s}  {'cert':>6s}")
    print("-" * 42)
    for t in key_tau:
        routed = gate_logits > t
        tpr_val = routed[cert_labels == 1].mean() if (cert_labels == 1).any() else 0
        fpr_val = routed[cert_labels == 0].mean() if (cert_labels == 0).any() else 0
        route_rate = routed.mean()
        cert_rate = (routed & (cert_labels == 1)).mean()
        print(f"{t:>8.1f}  {tpr_val:>6.4f}  {fpr_val:>6.4f}  {route_rate:>6.4f}  {cert_rate:>6.4f}")

    # ---------- plot ----------
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"gate (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="random (AUC=0.5)")

    plotted = set()
    for t in key_tau:
        routed = gate_logits > t
        tpr_val = routed[cert_labels == 1].mean() if (cert_labels == 1).any() else 0
        fpr_val = routed[cert_labels == 0].mean() if (cert_labels == 0).any() else 0
        ax.plot(fpr_val, tpr_val, "ro", markersize=5)
        # only annotate if this point is not overlapping a previous one
        key = (round(fpr_val, 3), round(tpr_val, 3))
        if key not in plotted:
            ax.annotate(f"tau={t}", (fpr_val, tpr_val), fontsize=7,
                        textcoords="offset points", xytext=(5, 3))
            plotted.add(key)

    ax.set_xlabel("False Positive Rate (route non-certifiable)")
    ax.set_ylabel("True Positive Rate (route certifiable)")
    ax.set_title(f"Gate ROC  (eps={eps:.5f}, n={n_total})")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    if args.output:
        out_path = args.output
    else:
        lbl = args.label or os.path.basename(run_dir or "ckpt").replace("/", "_")
        out_path = f"roc_gate_{lbl}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nplot saved: {out_path}")

    # bare version: the curve only, no title/tau/legend, for the thesis
    bare_path = os.path.splitext(out_path)[0] + "_bare.pdf"
    figb, axb = plt.subplots(1, 1, figsize=(6, 5))
    axb.plot(fpr, tpr, "b-", linewidth=2)
    axb.plot([0, 1], [0, 1], "k--", linewidth=1)
    axb.set_xlabel("False Positive Rate")
    axb.set_ylabel("True Positive Rate")
    axb.set_xlim(-0.02, 1.02)
    axb.set_ylim(-0.02, 1.02)
    axb.grid(True, alpha=0.3)
    figb.savefig(bare_path, bbox_inches="tight")
    print(f"bare plot saved: {bare_path}")


if __name__ == "__main__":
    main()

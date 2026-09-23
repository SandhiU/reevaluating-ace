#!/usr/bin/env python3
"""Standalone core-network evaluation (nat + PGD), bypassing ACE's main.py.

ACE's main.py --train-mode test doesn't work for EfficientNet because
UpscaleNet.determine_dims sets self.dims = None (networks.py:66).

This script uses ACE's own get_net() to load the model (so architecture
and normalization are identical), then evaluates nat + PGD directly.

Usage (HPC, ace env, GPU):
  cd ~/research/ACE && python ~/research/scripts/eval_core.py \
      --ckpt ./trained_models/EB-0_cifar10_adv_2_255.pt \
      --eps 0.00784313725 --label 2_255
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchattacks import PGD

# ACE imports - must run from ~/research/ACE or have it on PYTHONPATH
sys.path.insert(0, os.getcwd())
from utils import get_net
from loaders import get_loaders
from args_factory import get_args


def build_ace_args(eps):
    """Build minimal ACE args for CIFAR-10 test loader + EfficientNet-B0."""
    _argv = sys.argv[:]
    sys.argv = [_argv[0], "--net", "efficientnet-b0_pre",
                "--dataset", "cifar10", "--train-eps", str(eps), "--test-eps", str(eps)]
    try:
        args = get_args()
    finally:
        sys.argv = _argv
    # Override defaults
    args.dataset = "cifar10"
    args.net = "efficientnet-b0_pre"
    args.train_eps = eps
    args.test_eps = eps
    args.test_batch = 100
    args.train_batch = 100
    args.num_workers = 4
    args.no_cuda = False
    args.test_set = "test"
    args.load_model = None
    args.load_trunk_model = None
    args.load_branch_model = None
    args.load_gate_model = None
    args.cert_net_dim = None
    args.n_rand_proj = 50
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to ACE EfficientNet checkpoint")
    ap.add_argument("--eps", type=float, required=True, help="ε (e.g. 0.00784313725 for 2/255)")
    ap.add_argument("--label", default="core", help="Label for output")
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--pgd-steps", type=int, default=40)
    ap.add_argument("--pgd-step-size", type=float, default=0.035)
    ap.add_argument("--no-pgd", action="store_true", help="Skip PGD (nat only)")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== core eval: {a.label} ===")
    print(f"  ckpt: {a.ckpt}")
    print(f"  eps: {a.eps} ({a.eps*255:.1f}/255)")
    print(f"  device: {device}")

    if not os.path.exists(a.ckpt):
        sys.exit(f"checkpoint not found: {a.ckpt}")

    # Load model using ACE's own code - this ensures correct architecture + normalization
    args = build_ace_args(a.eps)
    _, _, test_loader, input_size, input_channel, n_class = get_loaders(args)
    net = get_net(device, args.dataset, args.net, input_size, input_channel, n_class,
                  load_model=a.ckpt)
    net.eval()

    # Verify it loaded something
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  model loaded: {n_params:,} parameters, dims={net.dims}")

    t0 = time.time()
    # Natural accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)
    nat_acc = correct / total
    print(f"  nat accuracy: {nat_acc:.4f} ({nat_acc*100:.1f}%)")
    print(f"  (paper reference: nat 95.1% at 2/255)")

    if not a.no_pgd:
        print(f"  running PGD ({a.pgd_steps} steps, step_size={a.pgd_step_size})...")
        atk = PGD(net, eps=a.eps, alpha=a.pgd_step_size, steps=a.pgd_steps, random_start=True)

        correct = 0
        total = 0
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            adv_inputs = atk(inputs, targets)
            with torch.no_grad():
                outputs = net(adv_inputs)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)
        pgd_acc = correct / total
        print(f"  PGD accuracy: {pgd_acc:.4f} ({pgd_acc*100:.1f}%)")
        print(f"  (paper reference: PGD 85.6% at 2/255)")

    elapsed = time.time() - t0
    print(f"  elapsed: {elapsed:.0f}s")
    print(f"=== {a.label} DONE ===")


if __name__ == "__main__":
    main()
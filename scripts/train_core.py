#!/usr/bin/env python3
"""Standalone adversarial training for EfficientNet-B0 core, bypassing main.py.

ACE's main.py doesn't work for EfficientNet (UpscaleNet.determine_dims sets
dims=None). This script loads the model via ACE's get_net() and trains it
with PGD adversarial training.

Recipe:
  - EfficientNet-B0, pretrained, CIFAR-10
  - PGD step size = eps/4 (Madry standard)
  - Adam, lr=1e-3, step decay (--lr-step, --lr-factor) or cosine
  - --nat-factor blends clean CE loss with adversarial CE loss:
      loss = (1 - nat_factor) * adv_loss + nat_factor * nat_loss
    nat_factor=0 (default) = pure adversarial; 0.5 = balanced (recommended for eps>=8/255)
  - Recommended: 60 epochs, step-decay 0.5 every 5, 10 PGD steps

Usage (HPC, ace env, GPU):
  cd ~/research/ACE && python ~/research/scripts/train_core.py \
      --exp-name core_adv_8_255 --train-eps 0.03137254901 \
      --nat-factor 0.5 --pgd-steps-train 10 --epochs 60
"""
import argparse
import json
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torchattacks import PGD

sys.path.insert(0, os.getcwd())
from utils import get_net
from loaders import get_loaders
from args_factory import get_args


def build_args(train_eps):
    _argv = sys.argv[:]
    sys.argv = [_argv[0], "--net", "efficientnet-b0_pre", "--dataset", "cifar10",
                "--train-eps", str(train_eps), "--test-eps", str(train_eps)]
    try:
        args = get_args()
    finally:
        sys.argv = _argv
    args.dataset = "cifar10"
    args.net = "efficientnet-b0_pre"
    args.train_eps = train_eps
    args.test_eps = train_eps
    args.train_batch = 50
    args.test_batch = 50
    args.num_workers = 4
    args.no_cuda = False
    args.test_set = "test"
    args.cert_net_dim = None
    args.n_rand_proj = 50
    args.load_model = None
    args.load_trunk_model = None
    args.load_branch_model = None
    args.load_gate_model = None
    return args


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-name", default="core_adv_8_255")
    ap.add_argument("--exp-id", type=int, default=1)
    ap.add_argument("--train-eps", type=float, default=0.03137254901, help="8/255")
    ap.add_argument("--test-eps", type=float, default=0.03137254901, help="8/255")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-step", type=int, default=5, help="step decay period (epochs)")
    ap.add_argument("--lr-factor", type=float, default=0.5, help="step decay multiplier")
    ap.add_argument("--lr-schedule", default="step", choices=["step", "cosine"],
                    help="LR schedule: step (halve every --lr-step epochs) or cosine")
    ap.add_argument("--nat-factor", type=float, default=0.0,
                    help="Blend factor for clean loss: loss = (1-f)*adv + f*nat. 0=pure adv, 0.5=balanced")
    ap.add_argument("--pgd-steps-train", type=int, default=20)
    ap.add_argument("--pgd-steps-test", type=int, default=40)
    ap.add_argument("--pgd-step-size", type=float, default=None,
                    help="PGD step size in pixel space. Default: eps/4 (Madry standard).")
    ap.add_argument("--pgd-step-size-test", type=float, default=0.035,
                    help="PGD step size for eval in pixel space; matches ACE's test-att-step-size")
    ap.add_argument("--l1-reg", type=float, default=1e-5)
    ap.add_argument("--test-freq", type=int, default=5)
    ap.add_argument("--save-dir", default=None)
    a = ap.parse_args()

    # Madry standard: alpha = eps/4. (A bare 0.25 here was ~8x eps at 8/255, so every
    # step jumped to the corner of the eps-ball and the attack degenerated.)
    if a.pgd_step_size is None:
        a.pgd_step_size = a.train_eps / 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== core training: {a.exp_name} ===")
    print(f"  train_eps: {a.train_eps} ({a.train_eps*255:.1f}/255)")
    print(f"  test_eps: {a.test_eps} ({a.test_eps*255:.1f}/255)")
    print(f"  pgd_step_size (train): {a.pgd_step_size:.6f} ({a.pgd_step_size*255:.2f}/255)")
    print(f"  pgd_step_size (test):  {a.pgd_step_size_test}")
    print(f"  pgd_steps: train={a.pgd_steps_train}, test={a.pgd_steps_test}")
    print(f"  epochs: {a.epochs}")
    print(f"  device: {device}")

    # Load model and data via ACE
    args = build_args(a.train_eps)
    _, train_loader, test_loader, input_size, input_channel, n_class = get_loaders(args)
    model = get_net(device, args.dataset, args.net, input_size, input_channel, n_class)
    model.train()

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=a.lr)
    if a.lr_schedule == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs, eta_min=1e-6)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=a.lr_step, gamma=a.lr_factor)
    criterion = nn.CrossEntropyLoss()

    # PGD attacker for training
    atk_train = PGD(model, eps=a.train_eps, alpha=a.pgd_step_size,
                    steps=a.pgd_steps_train, random_start=True)

    # Save directory
    timestamp = int(time.time())
    if a.save_dir is None:
        a.save_dir = f"models_new/cifar10/{a.exp_name}/{a.exp_id}/efficientnet-b0_pre_{a.train_eps:.5f}/{timestamp}"
    os.makedirs(a.save_dir, exist_ok=True)

    # Save args
    with open(os.path.join(a.save_dir, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=2)

    print(f"  save_dir: {a.save_dir}")

    best_nat = 0
    for epoch in range(a.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        t0 = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Generate adversarial examples
            adv_inputs = atk_train(inputs, targets)

            optimizer.zero_grad()

            # Adversarial loss
            adv_outputs = model(adv_inputs)
            adv_loss = criterion(adv_outputs, targets)

            # Natural loss (when nat_factor > 0)
            if a.nat_factor > 0:
                nat_outputs = model(inputs)
                nat_loss = criterion(nat_outputs, targets)
                loss = (1 - a.nat_factor) * adv_loss + a.nat_factor * nat_loss
            else:
                loss = adv_loss

            # L1 regularization
            l1_loss = 0
            for param in model.parameters():
                l1_loss += torch.norm(param, 1)
            loss += a.l1_reg * l1_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # Track accuracy on adversarial examples
            correct += (adv_outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)

        scheduler.step()
        train_acc = correct / total
        elapsed = time.time() - t0

        # Evaluate
        if (epoch + 1) % a.test_freq == 0 or epoch == a.epochs - 1:
            nat_acc = evaluate(model, test_loader, device)
            print(f"  epoch {epoch+1:3d}/{a.epochs}: loss={total_loss/len(train_loader):.4f} "
                  f"train_acc={train_acc:.4f} nat_acc={nat_acc:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.6f} ({elapsed:.0f}s)")

            if nat_acc > best_nat:
                best_nat = nat_acc
                ckpt_path = os.path.join(a.save_dir, "best_model.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"    saved best model (nat={nat_acc:.4f})")
        else:
            print(f"  epoch {epoch+1:3d}/{a.epochs}: loss={total_loss/len(train_loader):.4f} "
                  f"train_acc={train_acc:.4f} lr={scheduler.get_last_lr()[0]:.6f} ({elapsed:.0f}s)")

    # Save final model
    final_path = os.path.join(a.save_dir, "final_model_state.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n=== training done ===")
    print(f"  best nat: {best_nat:.4f}")
    print(f"  final: {final_path}")
    print(f"\nEvaluate with:")
    print(f"  python ~/research/scripts/eval_core.py \\")
    print(f"      --ckpt {final_path} --eps {a.test_eps} --label {a.exp_name}")


if __name__ == "__main__":
    main()

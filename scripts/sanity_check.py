#!/usr/bin/env python3
"""Sanity check: train a small IBP model on a CIFAR-10 subset.

Trains ~5 epochs on 1000 samples to verify CTRAIN works end-to-end.
"""

import os

import torch
from CTRAIN.data_loaders import load_cifar10
from CTRAIN.model_definitions import CNN7_Shi
from CTRAIN.model_wrappers import ShiIBPModelWrapper
from helpers.logging import ExperimentLogger

BATCH_SIZE = 128
EPS = 2 / 255
NUM_EPOCHS = 5
TRAIN_SUBSET = 1000
TEST_SUBSET = 200
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)

log = ExperimentLogger(
    name="sanity_ibp",
    extra_meta={
        "method": "ShiIBP",
        "eps": str(EPS),
        "epochs": NUM_EPOCHS,
        "train_samples": TRAIN_SUBSET,
        "device": DEVICE,
    },
)

train_loader, test_loader = load_cifar10(
    batch_size=BATCH_SIZE, val_split=False, data_root="./data",
)

def _subset_loader(loader, n):
    collected = []
    total = 0
    for x, y in loader:
        collected.append((x, y))
        total += x.size(0)
        if total >= n:
            break
    if not collected:
        return loader
    subset_x = torch.cat([c[0] for c in collected])[:n]
    subset_y = torch.cat([c[1] for c in collected])[:n]
    sub_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(subset_x, subset_y),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    # Preserve metadata that CTRAIN loaders attach
    for attr in ["normalised", "mean", "std", "min", "max"]:
        if hasattr(loader, attr):
            setattr(sub_loader, attr, getattr(loader, attr))
    return sub_loader

train_sub = _subset_loader(train_loader, TRAIN_SUBSET)
test_sub = _subset_loader(test_loader, TEST_SUBSET)

input_shape = [3, 32, 32]
model = CNN7_Shi(in_shape=input_shape)

wrapped_model = ShiIBPModelWrapper(
    model=model, input_shape=input_shape,
    eps=EPS, num_epochs=NUM_EPOCHS, device=DEVICE,
    warm_up_epochs=0, ramp_up_epochs=3,
)

wrapped_model.train_model(train_sub)

std_acc, cert_acc, adv_acc = wrapped_model.evaluate(test_sub)

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
ckpt_path = os.path.join(checkpoint_dir, "sanity_ibp.pt")
torch.save(wrapped_model.state_dict(), ckpt_path)

log.log_final({
    "std_acc": std_acc,
    "cert_acc": cert_acc,
    "adv_acc": adv_acc,
    "checkpoint": ckpt_path,
})
log.close()

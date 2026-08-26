#!/usr/bin/env python3
import os
import torch

ACE_DIR = os.path.expanduser("~/research/ACE/ace_core/trained_models")
RUNS = os.path.expanduser("~/research/ctrain-work/runs")

TEMPLATES = {
    "2_255": os.path.join(ACE_DIR, "C3_ACE_Net_IBP_cert_cifar10_2_255.pt"),
    "8_255": os.path.join(ACE_DIR, "C3_ACE_Net_IBP_cert_cifar10_8_255.pt"),
}
MATRIX = [
    ("sabr",    0.03876960, "2_255", "SABR"),
    ("sabr",    0.15507840, "8_255", "SABR"),
    ("mtl_ibp", 0.03876960, "2_255", "MTL"),
    ("mtl_ibp", 0.15507840, "8_255", "MTL"),
    ("ibp",     0.03876960, "2_255", "IBP"),
    ("ibp",     0.15507840, "8_255", "IBP"),
]
SRC_LAYERS = [0, 2, 4, 7, 9]


def bridge_one(method, eps, eps_sfx, mname):
    tpl = torch.load(TEMPLATES[eps_sfx], map_location="cpu")
    path = os.path.join(RUNS, f"{method}_eps{eps:.8f}", "final_model_state.pt")
    assert os.path.exists(path), f"missing our model: {path}"
    ours = torch.load(path, map_location="cpu")
    branch = "branchNet_0"
    n = 0
    CONV, LINEAR = {0, 2, 4}, {7, 9}
    for i in SRC_LAYERS:
        d = i + 1
        w, b = ours[f"{i}.weight"], ours[f"{i}.bias"]
        sub = "conv" if i in CONV else "linear"
        for dst in (f"{branch}.blocks.layers.{d}.weight", f"{branch}.blocks.layers.{d}.{sub}.weight"):
            assert tuple(tpl[dst].shape) == tuple(w.shape), f"shape {dst}"
            tpl[dst] = w; n += 1
        for dst in (f"{branch}.blocks.layers.{d}.bias", f"{branch}.blocks.layers.{d}.{sub}.bias",
                    f"{branch}.blocks.layers.{d}.b"):
            assert tuple(tpl[dst].shape) == tuple(b.shape), f"shape {dst}"
            tpl[dst] = b; n += 1
    sel = os.path.join(ACE_DIR, f"C3_ACE_Net_{mname}_cert_cifar10_{eps_sfx}_v2.pt")
    torch.save(tpl, sel)
    ent = {k: v for k, v in tpl.items() if not k.startswith("gateNet_0.")}
    entp = os.path.join(ACE_DIR, f"C3_ACE_Entropy_{mname}_cert_cifar10_{eps_sfx}_v2.pt")
    torch.save(ent, entp)
    print(f"OK {method} {eps_sfx}: {n} tensors -> {os.path.basename(sel)} ({len(tpl)} keys), "
          f"{os.path.basename(entp)} ({len(ent)} keys)")


for row in MATRIX:
    bridge_one(*row)

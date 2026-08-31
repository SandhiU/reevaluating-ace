#!/bin/bash
# apply_patches.sh — patches third-party packages after pip install.
# Run from the repo root with the `ace` conda env active:
#   conda activate ace && bash apply_patches.sh
#
# Fixes:
#   1. robustness      — torchvision.models.utils removed in newer torchvision
#   2. ACE utils.py    — load_net_state shape-compare bug (gate loading)
#   3. ACE relaxed_networks.py — CombinedNetwork has no n_class (SeqNet doesn't define
#                                one → entropy gate crashes); derive the net's real output dim
#   4. ACE relaxed_networks.py — CombinedNetwork.forward() misses kappa kwarg
#                                (trainer.py:230 passes it => entropy co-training
#                                 crashes with TypeError; also silently kills
#                                 trunk certification in ai_cert_sample)

set -e
PY="${PY:-python}"
ACE_DIR="${ACE_DIR:-$HOME/research/ACE}"

echo "=== 1/4 robustness: torchvision.models.utils -> torch.hub ==="
ROBUSTNESS_DIR=$($PY -c "import robustness, os; print(os.path.dirname(robustness.__file__))" 2>/dev/null || true)
if [ -n "$ROBUSTNESS_DIR" ]; then
    FILES=$(grep -rl "from torchvision.models.utils import load_state_dict_from_url" "$ROBUSTNESS_DIR" 2>/dev/null || true)
    for f in $FILES; do
        sed -i 's/from torchvision.models.utils import load_state_dict_from_url/from torch.hub import load_state_dict_from_url/g' "$f"
        echo "  patched $f"
    done
    [ -z "$FILES" ] && echo "  already patched or no matches"
else
    echo "  robustness not installed — skipped"
fi

echo "=== 2/4 ACE utils.py: load_net_state shape-compare bug ==="
UTILS="$ACE_DIR/utils.py"
if [ -f "$UTILS" ]; then
    if grep -q "v_load.numel() == state_dict_new\[k_load\].numel()" "$UTILS"; then
        echo "  already patched"
    else
        $PY - "$UTILS" << 'EOF'
import sys
path = sys.argv[1]
src = open(path).read()
old = "if k_load in state_dict_new and all(v_load.squeeze().shape == np.array(state_dict_new[k_load].squeeze().shape)):"
new = "if k_load in state_dict_new and v_load.numel() == state_dict_new[k_load].numel():"
assert old in src, "pattern not found in utils.py — check ACE version"
open(path, "w").write(src.replace(old, new))
print("  patched")
EOF
    fi
else
    echo "  $UTILS not found — skipped"
fi

echo "=== 3/4 ACE relaxed_networks.py: CombinedNetwork.n_class ==="
# get_LiRPA_losses() and get_abs_loss() read self.net.n_class, but only myNet defines it —
# a plain SeqNet (e.g. the entropy gate: branch blocks + Entropy layer) does not, so those
# calls raise AttributeError. Derive the output dim the way SeqNet.verify does:
# net.n_class if present, else blocks[-1].out_features (the Entropy layer sets it to 1).
# The value must be exact: get_abs_loss branches on `n_class > 1`, and a binary gate net sent
# down the multi-class branch gets BCE(-lb, y) instead of BCE(worst-case logit, y), i.e. an
# inverted certified loss that trains the gate to be un-certifiable.
RELAXED="$ACE_DIR/relaxed_networks.py"
if [ -f "$RELAXED" ]; then
    if grep -q "self.n_class = None if net is None" "$RELAXED"; then
        echo "  already patched"
    else
        $PY - "$RELAXED" << 'EOF'
import sys
path = sys.argv[1]
src = open(path).read()

plain = "        self.net = net\n        self.relaxed_net = relaxed_net"
patched = ("        self.net = net\n"
           "        # Output dim of the wrapped net: myNet sets n_class, SeqNet does not, so fall\n"
           "        # back to the last block's out_features (Entropy sets out_features = 1).\n"
           "        self.n_class = None if net is None else getattr(net, 'n_class', None)\n"
           "        if self.n_class is None and net is not None:\n"
           "            self.n_class = getattr(net.blocks[-1], 'out_features', 1)\n"
           "        if self.n_class is None:\n"
           "            self.n_class = 1\n"
           "        self.relaxed_net = relaxed_net")

if plain not in src:
    raise SystemExit("  __init__ pattern not found - check ACE version")
src = src.replace(plain, patched, 1)

src = src.replace("n_class = self.net.n_class", "n_class = self.n_class", 1)
src = src.replace('if domain in ["box","hbox"] and self.net.n_class > 1:',
                  'if domain in ["box","hbox"] and self.n_class > 1:', 1)
assert "self.net.n_class" not in src, "unexpected remaining self.net.n_class"

open(path, "w").write(src)
print("  patched")
EOF
    fi
else
    echo "  $RELAXED not found — skipped"
fi

echo "=== 4/4 ACE relaxed_networks.py: forward()/get_abs_loss() kappa kwarg ==="
# trainer.py:230 calls cnet(..., kappa=kappa) for every cnet in cnets[1:] — that
# list is non-empty ONLY on the entropy co-training path ([branch_cnet, gate_cnet]),
# which is why SelNet runs are fine and every entropy run dies with
#   TypeError: CombinedNetwork.forward() got an unexpected keyword argument 'kappa'
# deepTrunk_main.ai_cert_sample also calls get_abs_loss(..., kappa=1) for the trunk,
# swallowed by a bare except -> "Certification of trunk failed critically".
# kappa is only used for nat_factor, which train() already applies outside, so
# accepting + ignoring it reproduces the intended behaviour.
if [ -f "$RELAXED" ]; then
    if grep -q "kappa=None" "$RELAXED"; then
        echo "  already patched"
    else
        $PY - "$RELAXED" << 'EOF'
import sys
path = sys.argv[1]
src = open(path).read()

old1 = "    def get_abs_loss(self, inputs, targets, eps, domain, threshold_min, beta=None):"
new1 = "    def get_abs_loss(self, inputs, targets, eps, domain, threshold_min, beta=None, kappa=None):"
assert old1 in src, "get_abs_loss signature not found — check ACE version"
src = src.replace(old1, new1, 1)

old2 = "                   is_train=False, relu_stable_type=\"tight\", robust_loss_mode=None, train_mode=None, domains=None,beta=1):"
new2 = "                   is_train=False, relu_stable_type=\"tight\", robust_loss_mode=None, train_mode=None, domains=None,beta=1, kappa=None):"
assert old2 in src, "forward signature not found — check ACE version"
src = src.replace(old2, new2, 1)

open(path, "w").write(src)
print("  patched")
EOF
    fi
else
    echo "  $RELAXED not found — skipped"
fi

echo "=== apply_patches.sh DONE ==="

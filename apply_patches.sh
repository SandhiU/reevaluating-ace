#!/bin/bash
# apply_patches.sh — patches third-party packages after pip install.
# Run from the repo root with the `ace` conda env active:
#   conda activate ace && bash apply_patches.sh
#
# Fixes:
#   1. robustness      — torchvision.models.utils removed in newer torchvision
#   2. ACE utils.py    — load_net_state shape-compare bug (gate loading)
#   3. ACE relaxed_networks.py — CombinedNetwork.n_class missing (entropy gate eval)

set -e
PY=python
ACE_DIR="./ACE"

echo "=== 1/3 robustness: torchvision.models.utils -> torch.hub ==="
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

echo "=== 2/3 ACE utils.py: load_net_state shape-compare bug ==="
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

echo "=== 3/3 ACE relaxed_networks.py: CombinedNetwork.n_class (entropy gate) ==="
RELAXED="$ACE_DIR/relaxed_networks.py"
if [ -f "$RELAXED" ]; then
    if grep -q "self.n_class = net.final_shape" "$RELAXED"; then
        echo "  already patched"
    else
        $PY - "$RELAXED" << 'EOF'
import sys
path = sys.argv[1]
src = open(path).read()

# 1) add n_class attribute in __init__ (after self.net = net)
old1 = "        self.net = net\n        self.relaxed_net = relaxed_net"
new1 = ("        self.net = net\n"
        "        self.n_class = net.final_shape[1] if hasattr(net, 'final_shape') and net.final_shape is not None else 10\n"
        "        self.relaxed_net = relaxed_net")
assert old1 in src, "init pattern not found"
src = src.replace(old1, new1, 1)

# 2) use self.n_class instead of self.net.n_class (2 occurrences)
old2 = "n_class = self.net.n_class"
assert old2 in src, "get_LiRPA_losses pattern not found"
src = src.replace(old2, "n_class = self.n_class  # self.net.n_class", 1)

old3 = 'if domain in ["box","hbox"] and self.net.n_class > 1:'
assert old3 in src, "get_abs_loss pattern not found"
src = src.replace(old3, 'if domain in ["box","hbox"] and self.n_class > 1:', 1)

open(path, "w").write(src)
print("  patched")
EOF
    fi
else
    echo "  $RELAXED not found — skipped"
fi

echo "=== apply_patches.sh DONE ==="

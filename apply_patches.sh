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
#   5. ACE relaxed_networks.py + deepTrunk_networks.py: add the l_alpha (alpha-CROWN)
#                                cert domain, wired to the gate and the branch together
#   6. ACE layers.py: Normalization returns a 4D constant so auto_LiRPA's backward
#                                does not assert (operators/bivariate.py:_multiply_by_const)
#   7. auto_LiRPA operators/bivariate.py: promote any non-4D constant to 4D in the
#                                backward pass (general net for the same assert)
#   8. ACE relaxed_networks.py: put optimize/opt_steps at the TOP level of the LiRPA
#                                bound_opts (ACE nests them under 'relu', so the alpha
#                                optimizer never ran and l_alpha fell back to plain CROWN)
#   9. ACE relaxed_networks.py: l_alpha must call compute_bounds(method='alpha-crown');
#                                'backward'/'crown' is the plain-CROWN no-op

set -e
PY="${PY:-python}"
ACE_DIR="${ACE_DIR:-$HOME/research/ACE}"

echo "=== 1/9 robustness: torchvision.models.utils -> torch.hub ==="
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

echo "=== 2/9 ACE utils.py: load_net_state shape-compare bug ==="
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

echo "=== 3/9 ACE relaxed_networks.py: CombinedNetwork.n_class ==="
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

echo "=== 4/9 ACE relaxed_networks.py: forward()/get_abs_loss() kappa kwarg ==="
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

echo "=== 5/9 ACE alpha-CROWN domain (l_alpha) for branch and gate ==="
# Adds an alpha-CROWN domain to ACE's verifier. In ai_cert_sample the SAME domain is
# passed to gate_cnets and branch_cnets, so certifying with l_alpha runs alpha-CROWN on
# both. The optimizer is switched on by building the gate/branch cnets with
# bound_opts={'optimize': True, ...} when ALPHA_OPT is set, and by passing the opt
# kwarg to compute_bounds. Set ALPHA_OPT=1 (and ALPHA_OPT_STEPS, default 100) in the
# eval environment. This is what makes --cert-domain l_alpha usable from pipeline.sh.
if [ -f "$RELAXED" ]; then
    if grep -q "domain == 'l_alpha'" "$RELAXED"; then
        echo "  already patched"
    else
        $PY - "$RELAXED" << 'EOF'
import sys
path = sys.argv[1]
src = open(path).read()

# 1. route the new domain to the LiRPA path
old = '        elif domain in ["l_IBP", "l_CROWN", "l_CROWN-IBP", "lf_IBP", "lf_CROWN", "lf_CROWN-IBP"]:'
new = '        elif domain in ["l_IBP", "l_CROWN", "l_CROWN-IBP", "lf_IBP", "lf_CROWN", "lf_CROWN-IBP", "l_alpha"]:'
assert old in src, "get_abs_loss domain list not found - check ACE version"
src = src.replace(old, new, 1)

# 2. add the alpha-CROWN branch to get_LiRPA_losses, before the margin assembly
anchor = ('        if bound_upper:\n'
          '            ub_margin = ub if not domain == \'l_CROWN-IBP\' else torch.min(iub, cub)')
block = (
'        elif domain == \'l_alpha\':\n'
'            import inspect as _inspect, os as _os\n'
'            _kw = {}\n'
'            _params = _inspect.signature(model.compute_bounds).parameters\n'
'            for _name in ("iteration", "opt_steps"):\n'
'                if _name in _params:\n'
'                    _kw[_name] = int(_os.environ.get("ALPHA_OPT_STEPS", "100"))\n'
'                    break\n'
'            lb, ub = model(method_opt="compute_bounds", x=x, IBP=False, C=c, method=\'backward\',\n'
'                           bound_lower=bound_lower, bound_upper=bound_upper,\n'
'                           final_node_name=final_node_name, no_replicas=True, **_kw)\n'
+ anchor)
assert anchor in src, "margin-assembly anchor not found - check ACE version"
src = src.replace(anchor, block, 1)

open(path, "w").write(src)
print("  patched")
EOF
    fi
else
    echo "  $RELAXED not found - skipped"
fi

# deepTrunk_networks.py: build gate/branch cnets with the optimizer when ALPHA_OPT is set
DTN="$ACE_DIR/deepTrunk_networks.py"
if [ -f "$DTN" ]; then
    if grep -q "ALPHA_OPT" "$DTN"; then
        echo "  deepTrunk_networks already patched"
    else
        $PY - "$DTN" << 'EOF'
import sys
path = sys.argv[1]
src = open(path).read()
if "\nimport os" not in src and src.startswith("import") is False:
    src = "import os\n" + src
elif "\nimport os" not in src:
    src = src.replace("\n", "\nimport os\n", 1)

trunk_anchor = "        self.trunk_cnet = CombinedNetwork.get_cnet(self.trunk_net, device, lossFn, evalFn, n_rand_proj, no_r_net=True, input_channels=self.input_channel)"
trunk_new = ("        _alpha_opts = None\n"
             "        if os.environ.get(\"ALPHA_OPT\"):\n"
             "            _alpha_opts = {\"optimize\": True, \"opt_steps\": int(os.environ.get(\"ALPHA_OPT_STEPS\", \"100\"))}\n"
             + trunk_anchor)
assert trunk_anchor in src, "trunk get_cnet not found"
src = src.replace(trunk_anchor, trunk_new, 1)

src = src.replace("                                                                 threshold_min=self.threshold[exit_idx])",
                  "                                                                 threshold_min=self.threshold[exit_idx], lirpa_bound_opts=_alpha_opts)", 1)
src = src.replace("            self.branch_cnets[exit_idx] = CombinedNetwork.get_cnet(self.branch_nets[exit_idx], device, lossFn, evalFn,\n                                                                   n_rand_proj)",
                  "            self.branch_cnets[exit_idx] = CombinedNetwork.get_cnet(self.branch_nets[exit_idx], device, lossFn, evalFn,\n                                                                   n_rand_proj, lirpa_bound_opts=_alpha_opts)", 1)

open(path, "w").write(src)
print("  patched")
EOF
    fi
else
    echo "  $DTN not found - skipped"
fi

echo "=== 6/9 ACE layers.py: 4D constant for auto_LiRPA backward ==="
# get_mean_sigma returns mean/sigma shaped (C,1,1) (ndim 3). Normalization.forward does
# (x - mean) / sigma, and auto_LiRPA's backward asserts the divisor const is 4D
# (operators/bivariate.py:_multiply_by_const). The HybridZonotope (box/zono) path never
# hits this, only the l_* (auto_LiRPA) path does. Add the batch dim so the constant is
# (1,C,1,1); it is constant-folded, so the math is unchanged.
LAYERS="$ACE_DIR/layers.py"
if [ -f "$LAYERS" ]; then
    if grep -q "mean.view(1" "$LAYERS"; then
        echo "  already patched"
    else
        $PY - "$LAYERS" << 'EOF'
import sys
path = sys.argv[1]
src = open(path).read()
old = "        return (x - self.mean) / self.sigma"
new = ("        # auto_LiRPA's backward asserts a 4D constant; mean/sigma are (C,1,1).\n"
       "        m = self.mean.view(1, *self.mean.shape)\n"
       "        s = self.sigma.view(1, *self.sigma.shape)\n"
       "        return (x - m) / s")
assert old in src, "Normalization.forward pattern not found - check ACE version"
src = src.replace(old, new, 1)
open(path, "w").write(src)
print("  patched")
EOF
    fi
else
    echo "  $LAYERS not found - skipped"
fi

echo "=== 7/9 auto_LiRPA: promote non-4D constants in the backward pass ==="
# ACE's Normalization divides by a (C,1,1) constant, and auto_LiRPA's bivariate backward
# asserts the constant is 4D (operators/bivariate.py:_multiply_by_const). Patch 6 fixes the
# Normalization at the source; this is a general net that promotes any lower-rank constant
# reaching that code to 4D (prepend batch/channel dims, which is the pattern the broadcast
# expects). Only fires on the l_* (auto_LiRPA) path.
LIRPA_DIR=$($PY -c "import auto_LiRPA, os; print(os.path.dirname(auto_LiRPA.__file__))" 2>/dev/null || true)
if [ -n "$LIRPA_DIR" ]; then
    BIV="$LIRPA_DIR/operators/bivariate.py"
    if [ -f "$BIV" ]; then
        if grep -q "4 - const.ndim" "$BIV"; then
            echo "  already patched"
        else
            $PY - "$BIV" << 'EOF'
import re, sys
path = sys.argv[1]
src = open(path).read()
m = re.search(r"^(\s*)assert isinstance\(const, torch\.Tensor\) and const\.ndim == 4\s*$", src, re.M)
if not m:
    raise SystemExit("  assert not found - check auto_LiRPA version")
ind = m.group(1)
new = (f"{ind}if isinstance(const, torch.Tensor) and const.ndim != 4:\n"
       f"{ind}    const = const.reshape((1,) * (4 - const.ndim) + tuple(const.shape))\n"
       f"{ind}assert isinstance(const, torch.Tensor) and const.ndim == 4")
src = src[:m.start()] + new + src[m.end():]
open(path, "w").write(src)
print("  patched")
EOF
        fi
    else
        echo "  $BIV not found - skipped"
    fi
else
    echo "  auto_LiRPA not installed - skipped"
fi

echo "=== 8/9 ACE relaxed_networks.py: top-level auto_LiRPA optimize for alpha-CROWN ==="
# CombinedNetwork builds the LiRPA module with bound_opts={'relu': bound_opts}, i.e. ACE's
# dict is the RELU options. So 'optimize'/'opt_steps' never reach the top level, auto_LiRPA
# never runs the alpha optimizer, and l_alpha silently degrades to plain CROWN (fast, no
# better than box). Inject optimize/opt_steps at the top level when ALPHA_OPT is set.
if [ -f "$RELAXED" ]; then
    if grep -q "_alpha_extra" "$RELAXED"; then
        echo "  already patched"
    else
        $PY - "$RELAXED" << 'EOF'
import sys
path = sys.argv[1]
src = open(path).read()
if "\nimport os" not in src:
    src = "import os\n" + src
old1 = "            self.lirpa_net = BoundedModule(net, dummy_input, bound_opts={'relu': bound_opts}, device=device)"
new1 = ("            _alpha_extra = {}\n"
        "            if os.environ.get('ALPHA_OPT'):\n"
        "                _alpha_extra = {'optimize': True, 'opt_steps': int(os.environ.get('ALPHA_OPT_STEPS', '100'))}\n"
        "            self.lirpa_net = BoundedModule(net, dummy_input, bound_opts={'relu': bound_opts, **_alpha_extra}, device=device)")
assert old1 in src, "lirpa_net BoundedModule not found"
src = src.replace(old1, new1, 1)
old2 = "                                       bound_opts={'relu': bound_opts, 'loss_fusion': True}, device=device)"
new2 = "                                       bound_opts={'relu': bound_opts, 'loss_fusion': True, **_alpha_extra}, device=device)"
assert old2 in src, "lirpa_net_loss BoundedModule not found"
src = src.replace(old2, new2, 1)
open(path, "w").write(src)
print("  patched")
EOF
    fi
else
    echo "  $RELAXED not found - skipped"
fi

echo "=== 9/9 ACE relaxed_networks.py: l_alpha must call method='alpha-crown' ==="
# The optimizer is selected by the compute_bounds METHOD string, not by bound_opts alone
# (cert-column-explainer.md:140, measured: alpha-crown 2.67 s, 'crown' 0.04 s = no-op).
# Patch 5 used method='backward', which is the plain-CROWN no-op, so l_alpha ran plain CROWN.
# Switch the l_alpha branch to method='alpha-crown'.
if [ -f "$RELAXED" ]; then
    if grep -q "method='alpha-crown'" "$RELAXED"; then
        echo "  already patched"
    else
        $PY - "$RELAXED" << 'EOF'
import re, sys
path = sys.argv[1]
src = open(path).read()
pat = r"method='backward',(\s*bound_lower=bound_lower, bound_upper=bound_upper,\s*final_node_name=final_node_name, no_replicas=True, \*\*_kw\))"
m = re.search(pat, src)
assert m, "l_alpha compute_bounds call not found (is patch 5 applied?)"
src = src[:m.start()] + "method='alpha-crown'," + m.group(1) + src[m.end():]
open(path, "w").write(src)
print("  patched")
EOF
    fi
else
    echo "  $RELAXED not found - skipped"
fi

echo "=== apply_patches.sh DONE ==="

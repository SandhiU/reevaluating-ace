#!/usr/bin/env python3
"""ACE's certification-specification matrix, dependency-free.

`ace_c_spec` builds the C-matrix ACE's box verifier uses (networks.py:92-95,
get_c_mat): rows are logit_true - logit_k for every k != y, self-spec removed,
shape (B, n_class-1, n_class).

Why this is its own module: gen_labels.py imports train_branch at module level,
and that drags in CTRAIN -> the vendored
abCROWN -> a bare `from utils import expand_path`. Importing gen_labels therefore
pulled in the whole CTRAIN stack. gen_labels_eval.py also needs the ACE *net*
classes (deepTrunk_networks / loaders), and importing those caches ACE's own
`utils` under the name `utils`; when the CTRAIN import then runs its bare
`from utils import expand_path`, it finds ACE's cached utils (no expand_path) ->
ImportError. Keeping this pure-torch helper CTRAIN-free lets gen_labels_eval.py
import the C-matrix without ever touching train_branch/CTRAIN.

Only depends on torch. Do NOT add ACE or CTRAIN imports here.
"""
import torch


def ace_c_spec(y, n_class, device):
    """ACE's multi-class specification matrix (networks.py:92-95, get_c_mat).

    rows = logit_true - logit_k for every k, self-spec removed -> (B, n_class-1, n_class).
    """
    eye = torch.eye(n_class, dtype=torch.float32, device=device)
    c = eye[y].unsqueeze(1) - eye.unsqueeze(0)
    I = ~(y.unsqueeze(1) == torch.arange(n_class, device=device).unsqueeze(0))
    return c[I].view(y.size(0), n_class - 1, n_class)

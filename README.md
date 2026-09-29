# Re-evaluating ACE

Thesis project: re-evaluate ACE (Certify or Predict, Müller et al. 2021) with modern certified training methods (SABR, MTL-IBP) instead of the original IBP/CIBP/COLT branches.

## How ACE works

An ACE model has three parts:

| Part | Role | Who provides it |
|---|---|---|
| **Trunk** | Accurate network (EfficientNet-B0), not provably robust | We train it (`train_core`) |
| **Branch** | Certifiably robust network (myNet C3) | We train these with CTRAIN |
| **Selector** | Routes each input to trunk or branch | We train it on alpha-CROWN labels |

The selector decides per input: if the branch can certify it, use the branch; otherwise fall back to the trunk. Two selector types exist: **SelectionNet** (a small network) and **Entropy** (route when the branch's output entropy is low).

## Pipeline

```
  train_core   trunk, PGD adv training (EfficientNet-B0)     -> ACE/models_new/.../core_adv_*
  train        branch with CTRAIN, then convert to ACE fmt   -> runs/ then converted/
  labels       alpha-CROWN labels for the branch             -> labels/
  selector     gate trained on those labels                  -> ACE/models_new/...
  roc          gate ROC figure                               -> figures/
  eval         composition eval, sweep tau                   -> cert_log.csv per run
  released     evaluate a RELEASED ACE model (ibp|colt)      -> ACE/models_new/.../rel_*
  roc_released gate ROC of a RELEASED ACE model             -> figures/roc_released_*
  aggregate    collect cert_log.csv into one CSV             -> results/agg_<ver>.csv
```

Everything is driven by `pipeline.sh`. Conversion runs automatically at the end of `train`.

## Setup

```bash
conda create -n ace python=3.11 -y
conda activate ace
pip install CTRAIN
pip install git+https://github.com/Verified-Intelligence/auto_LiRPA.git
pip install -r requirements.txt
bash apply_patches.sh    # fixes third-party packages (e.g. robustness) and ACE
```

Two dependency notes: CTRAIN needs `scikit-learn==1.8` (smac incompatibility), and `apply_patches.sh` fixes the third-party and ACE bugs we hit (robustness import, ACE gate loading, ACE `n_class`, ACE `kappa` kwarg).

## Running the pipeline

```bash
bash pipeline.sh <stage> <target> [flags]

# targets: ibp | sabr | mtl-ibp | crown-ibp | all
# flags:   --ver VER  --in_ver VER  --gate-model METHOD  --in-gate-model VER
#          --sel | --ent   --eps 2_255|8_255|both
#          --verify box|alpha|alpha-gate|all   --trunk-model PATH   --dry-run
# released / roc_released targets are ibp | colt | all (not the CTRAIN methods)

bash pipeline.sh train_core --eps 8_255 --ver v5            # trunk
bash pipeline.sh train sabr --ver v5 --sel                  # branch (+ convert)
bash pipeline.sh labels all --ver v5                        # alpha-CROWN labels
bash pipeline.sh selector all --ver v5 --sel                # gates
bash pipeline.sh roc all --ver v5 --sel                     # ROC figures
bash pipeline.sh eval all --in_ver v5 --ver v5 --sel        # compositions
bash pipeline.sh released colt --eps 8_255 --ver v7         # released COLT 8/255
bash pipeline.sh roc_released colt --eps both --ver v7      # released COLT ROCs
bash pipeline.sh aggregate --in_ver v5                      # CSV of all eval runs

# single model, or a cross-gate pairing:
bash pipeline.sh train sabr --ver v5 --eps 2_255 --sel
bash pipeline.sh eval mtl-ibp --gate-model ibp --in_ver v5 --in-gate-model v5 --ver v5 --sel

# whole pipeline in one go (submit everything, then walk away):
bash pipeline.sh all ibp --ver test-22-09-2026-01 --in_ver v5
bash pipeline.sh all all --ver v5

bash pipeline.sh status v5
```

The version tag is any string, and it flows through every stage. Each stage reads `--in_ver` and writes `--ver`, so a full run with one tag is self-consistent, and a rerun can mix in an older version for any input.

## Layout

```
research/
├── ACE/            ACE codebase (clone; not tracked)
├── data/           CIFAR-10
├── scripts/        our scripts (this file lives in code/)
├── runs/           CTRAIN branch outputs          runs/<method>_<eps>_<ver>/
├── converted/      branches in ACE format         converted/<method>_<eps>_<ver>_branch.pt
├── labels/         alpha-CROWN selector targets   labels/labels_alpha_crown_<method>_<eps>_<ver>.csv
├── figures/        ROC and other figures
└── results/        aggregated results
```

`config.sh` holds all shared paths, the epsilon values, the tau grids, the trunks, and `branch_hp()` (the branch hyperparameters). It also carries the exp-id scheme and the model-discovery helpers used by the pipeline.

## Verification

`eval --verify` selects what the composition is certified with:

- `box` (default): ACE's box verifier, used by the released models and our box numbers.
- `alpha`: alpha-CROWN. In ACE the same domain is passed to the gate and the branch, so this certifies both with alpha-CROWN. Needs the alpha domain in ACE (`relaxed_networks.py`); see `apply_patches.sh`.
- `alpha-gate`: same wiring as `alpha` (alpha-CROWN on the gate as well), kept for clarity in the sweep.
- `all`: runs box and alpha.

Certified accuracy is a lower bound, so alpha-CROWN must be at least as high as box on the same functional. Our label path and ACE's box use the same C-matrix functional (`gen_labels.py --ace-spec`), which is what makes that monotonicity hold.

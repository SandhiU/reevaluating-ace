# Re-evaluating ACE

Thesis project: re-evaluate ACE (Certify or Predict, Müller et al. 2021) with modern certified training methods — SABR, MTL-IBP — instead of the original IBP/CIBP/COLT branches.

## How ACE works

An ACE model has three parts:

| Part | Role | Who provides it |
|---|---|---|
| **Trunk** | Accurate network (EfficientNet-B0), not provably robust | ACE's released checkpoint, frozen |
| **Branch** | Certifiably robust network (myNet C3) | We train these with CTRAIN |
| **Selector** | Routes each input to trunk or branch | Trained by ACE's own code |

The selector decides per input: if the branch can certify it, use the branch; otherwise fall back to the trunk. Two selector types exist: **SelectionNet** (a small network) and **Entropy** (route when the branch's output entropy is low).

## Pipeline (CTRAIN to ACE)

```
1. train_cert_net.py         train a branch with CTRAIN        -> runs/<method>_eps<eps>/
2. convert_format.py         CTRAIN checkpoint -> ACE format   -> converted/
3. train_selector.slurm      ACE's selector training           -> ACE/models_new/...
4. eval_ace.slurm            ACE's evaluation, sweep τ         -> cert_log.csv per run
5. aggregate_ace_results.py  collect results into a table      -> results/agg_<ver>.csv
```

**ACE already implements selector training and evaluation** (steps 3-4). Use the pipeline scripts above to run all models at once.

## Setup

```bash
conda create -n ace python=3.11 -y
conda activate ace
pip install CTRAIN
pip install git+https://github.com/Verified-Intelligence/auto_LiRPA.git
pip install -r requirements.txt
bash apply_patches.sh    # fixes third-party packages (e.g. robustness)
```

Two current dependency issues: CTRAIN needs `scikit-learn==1.8` (smac incompatibility), and `apply_patches.sh` fixes three third-party bugs:
1. `robustness` — `torchvision.models.utils` removed in newer torchvision
2. ACE `utils.py` — `load_net_state` shape-compare bug that breaks gate loading during selector training
3. ACE `relaxed_networks.py` — missing `n_class` on `CombinedNetwork` that breaks entropy-gate evaluation

## Quick start — pipeline scripts

Instead of constructing `sbatch` commands manually, use the convenience scripts in `scripts/`. They call the slurm files with correct parameters pre-set for all models.

```bash
# Run the full pipeline (pauses between stages)
bash pipeline.sh all v1

# Or run stages individually:
bash pipeline.sh train v1              # Stage 1: train cert networks
bash pipeline.sh convert               # Stage 2: convert to ACE format
bash pipeline.sh selector v1 --sel     # Stage 3: SelNet only
bash pipeline.sh selector v1 --ent     # Stage 3: Entropy only
bash pipeline.sh eval v1 --both        # Stage 4: evaluate everything

# Check what's done:
bash pipeline.sh status eval_v1

# Aggregate results into a CSV:
bash aggregate_results.sh v1           # → results/agg_v1.csv
bash aggregate_results.sh all          # → results/agg_all.csv
```

Stage 3 accepts a custom converted directory if you're reusing branches from a previous run:
```bash
bash 03_train_selector_all.sh v1 --both ~/research/converted
```

See `00_config.sh` for all shared paths and hyperparameters.

## Project structure

```
research/
├── ACE/                # ACE codebase (clone; not tracked in git)
├── data/               # CIFAR-10: cifar-10-batches-py + cifar-10-python.tar.gz
│                       # (tar tracked in git; extracted batches gitignored)
├── scripts/            # our scripts (this repo's content)
├── runs/               # trained branches (gitignored)
├── converted/          # branches in ACE format (gitignored)
├── results/            # aggregated results
├── requirements.txt
├── apply_patches.sh
└── README.md
```

### 1. Train a branch

```bash
sbatch --job-name=crown_2 train_cert_net.slurm \
    --method crown_ibp --eps 0.00784313725 --epochs 160 --lr-milestones 120,140
```

Methods: `ibp`, `sabr`, `mtl_ibp`, `crown_ibp`. Eps: `0.00784313725` (2/255) or `0.03137254901` (8/255).

### 2. Convert to ACE format

```bash
python convert_format.py --all
```

### 3. Train the selector (uses ACE's code)

```bash
sbatch train_selector.slurm --branch ../converted/sabr_2_255.pt \
    --gate-type net --eps 0.00784313725
```

`--gate-type entropy` for the entropy selector.

### 4. Evaluate (τ sweep)

```bash
sbatch eval_ace.slurm \
    --load-model <ACE model from step 3> \
    --gate-type net --gate-threshold 0.0,0.3,0.5,0.7,0.9 \
    --eps 0.00784313725
```

Two things that must match the model: `--gate-type` (`net`/`entropy`) and `--cert-domain` (`box` for IBP-style branches, `zono` for COLT). Entropy thresholds are negative (`-0.4` means "route if entropy ≤ 0.4").

### 5. Collect results

```bash
python aggregate_ace_results.py \
    --results-dir ~/research/ACE/models_new \
    --output ~/research/results/all_results.csv
```

#!/usr/bin/env python3
"""Audit alpha-CROWN (or any) selector label CSVs: coverage, balance, class balance.

Answers the current question directly: "what does the label dataset actually look
like, is it balanced?"  Reads the merged per-branch files
`labels_alpha_crown_<branch>[_<eps>].csv` (or the shards, if merged are absent) and
reports, per branch:

  * rows / index coverage / duplicates / gaps
  * positive rate (overall) -- this is the ACE gate-training `data_balance`
  * the ACE auto-balance weight it implies: pos_weight = (1-b)/(b+1e-3)
    (deepTrunk_main.py:206-207, update_loss_fn -> BCEWithLogits(pos_weight))
  * per-class positive rate and per-class counts (CIFAR-10 train order = sorted by
    class, so class = index // 5000; test order likewise for test-split files)
  * positive/negative counts per class -> is the *label* distribution class-skewed
    even though the *dataset* is balanced?
  * optional: agreement with a second label column (e.g. box/CROWN vs alpha),

Usage (HPC, ace env or any env with pandas):
  python label_stats.py ~/research/labels
  python label_stats.py ~/research/labels --eps 8_255
  python label_stats.py ~/research/labels --glob 'labels_alpha_crown_*_2_255*.csv'
  python label_stats.py ~/research/labels --split train     # 5000/class
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
PER_CLASS = {"train": 5000, "test": 1000}


def branch_name(fname, prefix):
    stem = os.path.basename(fname)
    stem = re.sub(r"\.csv$", "", stem)
    stem = re.sub(r"_shard\d+of\d+$", "", stem)
    return re.sub(r"^" + re.escape(prefix), "", stem)


def audit_one(path, per_class, split):
    df = pd.read_csv(path)
    if "label" not in df.columns:
        print(f"  SKIP {path}: no 'label' column")
        return None
    n = len(df)
    idx = df["index"]
    dup = int(idx.duplicated().sum())
    lo, hi = int(idx.min()), int(idx.max())
    exp = per_class * 10 * (2 if split == "all" else 1)
    missing = len(set(range(lo, hi + 1)) - set(idx))

    pos = df["label"].astype(bool)
    rate = float(pos.mean())
    # ACE auto-balance (deepTrunk_main.py:206-207)
    pw = (1 - rate) / (rate + 1e-3)

    print(f"\n  {os.path.basename(path)}")
    print(f"    rows={n}  index=[{lo},{hi}]  dupes={dup}  gaps_in_range={missing}"
          + (f"  (expected {exp} rows)" if exp else ""))
    print(f"    pos_rate={rate:.4f}  neg_rate={1-rate:.4f}  "
          f"implied pos_weight={pw:.3f}")
    if dup or (exp and n != exp):
        print("    NOTE: coverage problem -- do not train on this file as-is")

    cls = (idx // per_class).astype(int)
    ok = cls.between(0, 9)
    if ok.any():
        tab = pd.DataFrame({"cls": cls[ok], "pos": pos[ok].astype(int)})
        g = tab.groupby("cls")["pos"].agg(["count", "sum"])
        g["pos_rate"] = g["sum"] / g["count"]
        print(f"    per-class ({split} ordering, class = index // {per_class}):")
        for c, row in g.iterrows():
            print(f"      {c} {CIFAR10_CLASSES[c]:<10} n={int(row['count']):>5} "
                  f"pos={int(row['sum']):>5} ({row['pos_rate']:.3f})")
        print(f"    class skew of positives: min={g['pos_rate'].min():.3f} "
              f"max={g['pos_rate'].max():.3f} "
              f"spread={g['pos_rate'].max()-g['pos_rate'].min():.3f}")
    return rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir")
    ap.add_argument("--prefix", default="labels_alpha_crown_",
                    help="filename prefix for the merged files")
    ap.add_argument("--glob", default=None,
                    help="override file selection with this glob (in indir)")
    ap.add_argument("--eps", default=None,
                    help="only files matching this eps token, e.g. 2_255 or 8_255")
    ap.add_argument("--split", default="train", choices=["train", "test", "all"],
                    help="index ordering of the files (CIFAR-10 train = 5000/class)")
    ap.add_argument("--compare-col", default=None,
                    help="if present, report agreement with this column (e.g. label_crown)")
    a = ap.parse_args()

    if a.glob:
        files = sorted(glob.glob(os.path.join(a.indir, a.glob)))
    else:
        files = sorted(glob.glob(os.path.join(a.indir, a.prefix + "*.csv")))
        files = [f for f in files if "_shard" not in os.path.basename(f)]
        if not files:  # fall back to shards
            files = sorted(glob.glob(os.path.join(a.indir, a.prefix + "*_shard*.csv")))
    if a.eps:
        files = [f for f in files if a.eps in os.path.basename(f)]
    if not files:
        sys.exit(f"no label files found in {a.indir}")

    per_class = PER_CLASS["train" if a.split in ("train", "all") else "test"]
    print(f"# label stats: {len(files)} file(s) in {a.indir} (split={a.split})")
    print("# pos_rate = fraction of samples labelled certifiable (gate target = 1)")
    print("# pos_weight = the weight ACE would apply to positives when training the gate")

    rates = {}
    for f in files:
        b = branch_name(f, a.prefix)
        r = audit_one(f, per_class, a.split)
        if r is not None:
            rates[b] = r
        if a.compare_col:
            df = pd.read_csv(f)
            if a.compare_col in df.columns and "label" in df.columns:
                agree = float((df["label"].astype(bool) ==
                               df[a.compare_col].astype(bool)).mean())
                print(f"    agreement with {a.compare_col}: {agree:.4f}  "
                      f"(disagree={(1-agree)*100:.2f}%)")

    print("\n# summary (pos_rate / implied pos_weight)")
    for b, r in sorted(rates.items()):
        print(f"  {b:<20} {r:.4f}   {(1-r)/(r+1e-3):.3f}")
    if rates:
        v = list(rates.values())
        print(f"  spread across branches: {max(v)-min(v):.4f}")
    print("\n# dataset itself: CIFAR-10 train/test are class-balanced (5000/1000 per class);")
    print("# any per-class skew in the table above is in the LABELS, not the data.")


if __name__ == "__main__":
    main()

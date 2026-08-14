#!/usr/bin/env python3
"""
Pull ACE evaluation results from HPC and compare against published numbers.

For each experiment (1–40), find the cert_log.csv in:
  $BASE/{exp_name}/{exp_id}/efficientnet-b0_pre_*/*/cert_log.csv

Then read its last line (the "total" aggregated row) and extract:
  - nat_ok    → natural accuracy of the ACE system
  - ver_ok    → certified robust accuracy (zonotope or box domain)
  - pgd_ok    → PGD attack accuracy

Usage:
  python pull_results.py [--base BASE] [--csv EXPERIMENTS_CSV] [--hpc]

Options:
  --base BASE      Root directory for results (default: ~/research/ACE/ace_core/models_new/cifar10)
  --csv CSV         Path to experiments.csv (default: ./experiments.csv)
  --hpc             Print as a compact table, suitable for terminal output on HPC
  --markdown        Print as markdown table (for thesis notes)
"""

import argparse
import csv
import glob
import os
import sys
from typing import List, Optional, Tuple

# ── Published numbers from Müller et al. (2021) Table 1 ──
PUBLISHED = {
    # (dataset, eps, core_name, gate_type) → (nat_pct, cert_pct)
    # C2 = 2 conv layers, C3 = 3 conv layers, C5 = 5 conv layers
    # COLT and IBP are cert-network training methods
    # SelNet = learned selector, Entropy = entropy-based selector
    # eps values: 2/255 ≈ 0.00784, 8/255 ≈ 0.03137
    # Values stored as fractions (0-1 scale), matching cert_log.csv format
    (1, "cifar10", 0.00784313725, "net"):   (0.904, 0.205),   # C2 COLT SelNet ε=2/255
    (6, "cifar10", 0.00784313725, "net"):   (0.901, 0.275),   # C3 COLT SelNet ε=2/255
    (11, "cifar10", 0.00784313725, "net"):  (0.905, 0.185),   # C3 IBP SelNet ε=2/255
    (16, "cifar10", 0.00784313725, "entropy"): (0.934, 0.194), # C2 COLT Entropy ε=2/255
    (21, "cifar10", 0.00784313725, "entropy"): (0.902, 0.072),  # C2 IBP Entropy ε=2/255
    (26, "cifar10", 0.03137254901, "net"):  (0.771, 0.074),    # C3 COLT SelNet ε=8/255
    (31, "cifar10", 0.03137254901, "net"):  (0.831, 0.064),    # C3 IBP SelNet ε=8/255
    (36, "cifar10", 0.03137254901, "net"):  (0.801, 0.105),   # C5 C-IBP SelNet ε=8/255
}

# Map experiment ID to the "key" experiment ID in PUBLISHED
# (the first experiment in each threshold sweep)
FIRST_EXP_OF_CONFIG = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1,
    6: 6, 7: 6, 8: 6, 9: 6, 10: 6,
    11: 11, 12: 11, 13: 11, 14: 11, 15: 11,
    16: 16, 17: 16, 18: 16, 19: 16, 20: 16,
    21: 21, 22: 21, 23: 21, 24: 21, 25: 21,
    26: 26, 27: 26, 28: 26, 29: 26, 30: 26,
    31: 31, 32: 31, 33: 31, 34: 31, 35: 31,
    36: 36, 37: 36, 38: 36, 39: 36, 40: 36,
}


def parse_cert_csv(path: str) -> Optional[dict]:
    """Read the last line of cert_log.csv and return metrics as a dict."""
    try:
        with open(path) as f:
            lines = f.readlines()
        if len(lines) < 2:
            print(f"  WARN: {path}: only {len(lines)} line(s)", file=sys.stderr)
            return None
        last = lines[-1].strip()
        header = lines[0].strip().split(";")
        vals = last.split(";")
        if len(vals) != len(header):
            print(f"  WARN: {path}: column mismatch ({len(vals)} vs {len(header)})", file=sys.stderr)
            return None
        return dict(zip(header, vals))
    except FileNotFoundError:
        print(f"  MISSING: {path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  ERROR reading {path}: {e}", file=sys.stderr)
        return None


def fmt_pct(val_str: str) -> str:
    """Format a fraction (like '0.9023') as percentage string like '90.23'."""
    try:
        return f"{float(val_str) * 100:.1f}"
    except (ValueError, TypeError):
        return "???"


def load_experiments(csv_path: str) -> List[dict]:
    """Load experiment configs from experiments.csv, skipping comment lines."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            # Skip comment lines (e.g. "# CIFAR-10 eps=2/255, SelectionNet...")
            rid = row.get("id", "")
            if rid.startswith("#"):
                continue
            rows.append(row)
    return rows


def find_cert_path(base_dir: str, exp_name: str, exp_id: str) -> Optional[str]:
    """
    Find cert_log.csv for an experiment using glob.
    Pattern: {base_dir}/{exp_name}/{exp_id}/efficientnet-b0_pre_*/*/cert_log.csv
    (there's a hash/timestamp subdirectory between efficientnet-b0_pre_* and cert_log.csv)
    """
    pattern = os.path.join(base_dir, exp_name, exp_id, "efficientnet-b0_pre_*", "*", "cert_log.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[-1]  # take the last one (most recent if multiple runs)


def get_best_threshold_match(results: List[dict], paper_nat: float, paper_cert: float,
                              ver_col: str) -> Tuple[Optional[dict], float]:
    """
    Among multiple threshold runs for the same config, find which one matches
    the paper's (nat, cert) pair best. Uses Euclidean distance in (nat, cert) space.
    Returns (best_row, min_distance).
    """
    best_row = None
    best_dist = float("inf")
    for row in results:
        try:
            nat = float(row.get("nat_ok", 0))
            cert = float(row.get(ver_col, 0))
            # Both on 0-1 scale now
            dist = ((nat - paper_nat) ** 2 + (cert - paper_cert) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_row = row
        except (ValueError, TypeError):
            continue
    return best_row, best_dist


def main():
    parser = argparse.ArgumentParser(
        description="Pull ACE results and compare against published numbers."
    )
    parser.add_argument(
        "--base",
        default=os.path.expanduser("~/research/ACE/ace_core/models_new/cifar10"),
        help="Root directory for results (default: ~/research/ACE/ace_core/models_new/cifar10)",
    )
    parser.add_argument(
        "--csv",
        default=os.path.join(os.path.dirname(__file__) or ".", "experiments.csv"),
        help="Path to experiments.csv",
    )
    parser.add_argument("--hpc", action="store_true", help="Compact table output")
    parser.add_argument("--markdown", action="store_true", help="Markdown table output")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Write results to FILE.md (implies --markdown)")
    args = parser.parse_args()

    if not os.path.isdir(args.base):
        print(f"Error: base directory not found: {args.base}", file=sys.stderr)
        print("Tip: use --base to specify the correct path, or run this on the HPC node.",
              file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.csv):
        print(f"Error: experiments CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    experiments = load_experiments(args.csv)
    print(f"Loaded {len(experiments)} experiments from {args.csv}", file=sys.stderr)
    print(f"Searching for cert_log.csv in {args.base}/...\n", file=sys.stderr)

    # Collect results, grouped by config (first experiment ID)
    # Each entry: (exp_id, threshold_str, data_dict)
    config_results: dict[int, list[tuple[int, str, dict]]] = {}
    for exp in experiments:
        exp_id_str = exp["id"]
        exp_name = exp["name"]
        try:
            exp_id = int(exp_id_str)
        except ValueError:
            continue
        config_key = FIRST_EXP_OF_CONFIG.get(exp_id)
        if config_key is None:
            continue

        cert_path = find_cert_path(args.base, exp_name, exp_id_str)
        if cert_path is None:
            print(f"  MISSING: Exp #{exp_id} ({exp_name}): cert_log.csv not found", file=sys.stderr)
            continue

        data = parse_cert_csv(cert_path)
        if data is None:
            continue

        threshold = exp.get("threshold", "?")
        config_results.setdefault(config_key, []).append((exp_id, threshold, data))

    # ── Build Raw Data Table (all thresholds per config) ──
    config_keys = sorted(PUBLISHED.keys())
    raw_rows = []   # (exp_id, threshold, nat, cert, ver_col_name, cert_domain, to_cert_pct)
    match_rows = []  # (exp_id, best_τ, nat, cert, nat_paper, cert_paper, domain)

    for cfg_key in config_keys:
        exp_id, dataset, eps, gate_type = cfg_key
        paper_nat, paper_cert = PUBLISHED[cfg_key]
        results = config_results.get(exp_id, [])

        # --- Raw data rows (all thresholds) ---
        for eid, thresh, data in results:
            # Determine ver column
            if "ver_ok_zono" in data:
                vc = "ver_ok_zono"
            elif "ver_ok_box" in data:
                vc = "ver_ok_box"
            else:
                vc = "pgd_ok"
            nat_s = data.get("nat_ok", "0")
            ver_s = data.get(vc, "0")
            # Routing share: fraction sent to certified model
            to_cert = data.get("branch_0_p", "?")
            # Short ver col name for display
            vc_display = {"ver_ok_zono": "zono", "ver_ok_box": "box", "pgd_ok": "pgd"}.get(vc, vc)
            # Cert domain
            domain = "?"
            for exp in experiments:
                if int(exp["id"]) == eid:
                    domain = exp.get("cert_domain", "?")
                    break
            raw_rows.append((exp_id, thresh, fmt_pct(nat_s), fmt_pct(ver_s), vc_display,
                             fmt_pct(to_cert) if to_cert != "?" else "?", domain))

        # --- Match row (filtered, best τ) ---
        if not results:
            match_rows.append((exp_id, "—", "—", "—", fmt_pct(paper_nat), fmt_pct(paper_cert), "—"))
            continue

        # Determine ver column (from first result)
        first_row = results[0][2]
        if "ver_ok_zono" in first_row:
            ver_col = "ver_ok_zono"
        elif "ver_ok_box" in first_row:
            ver_col = "ver_ok_box"
        else:
            ver_col = "pgd_ok"

        # Cert domain
        cert_domain = "?"
        for exp in experiments:
            if int(exp["id"]) == results[0][0]:  # type: ignore[misc]
                cert_domain = exp.get("cert_domain", "?")
                break

        # Find best τ across ALL thresholds (no filtering — τ=0.0 is the natural decision boundary)
        best_row, best_dist = get_best_threshold_match(
            [d for _, _, d in results], paper_nat, paper_cert, ver_col
        )
        if best_row is None:
            match_rows.append((exp_id, "—", "—", "—", fmt_pct(paper_nat), fmt_pct(paper_cert), cert_domain))
            continue

        # Find which threshold
        best_threshold = "?"
        for eid, t, r in results:
            if r is best_row:
                best_threshold = t
                break

        nat_str = best_row.get("nat_ok", "0")
        ver_str = best_row.get(ver_col, "0")

        match_rows.append((
            exp_id, best_threshold, fmt_pct(nat_str), fmt_pct(ver_str),
            fmt_pct(paper_nat), fmt_pct(paper_cert), cert_domain
        ))

    # ── Render ──
    out_lines = []
    def blank_line():
        out_lines.append("")

    if args.output:
        args.markdown = True

    # ──── TABLE 1: Raw Data (all thresholds) ────
    if args.markdown:
        out_lines.append("## Table 1: Raw ACE Results (all thresholds)")
        out_lines.append("")
        out_lines.append("| Config | τ | Nat | Cert | Cert Col | →Cert% | Domain |")
        out_lines.append("|--------|---|-----|------|----------|--------|--------|")
        for r in raw_rows:
            cid, thresh, nat, cert, vc, to_cert, domain = r
            out_lines.append(f"| {cid} | {thresh} | {nat}% | {cert}% | {vc} | {to_cert}% | {domain} |")
    elif args.hpc or not args.markdown:
        out_lines.append(f"{'Config':<8} {'τ':<6} {'Nat':<7} {'Cert':<7} {'Col':<6} {'→Cert%':<8} {'Domain':<6}")
        out_lines.append("-" * 52)
        for r in raw_rows:
            cid, thresh, nat, cert, vc, to_cert, domain = r
            out_lines.append(f"{cid:<8} {thresh:<6} {nat:<7} {cert:<7} {vc:<6} {to_cert:<8} {domain:<6}")

    # ──── TABLE 2: Comparison (filtered, best τ) ────
    if args.markdown:
        blank_line()
        out_lines.append("## Table 2: ACE Comparison vs Published (best τ across all thresholds)")
        out_lines.append("")
        out_lines.append("| Config | Best τ | Nat (us) | Cert (us) | Nat (paper) | Cert (paper) | Domain |")
        out_lines.append("|--------|--------|----------|-----------|-------------|--------------|--------|")
        for r in match_rows:
            cid, thresh, nat_us, cert_us, nat_paper, cert_paper, domain = r
            out_lines.append(f"| {cid} | {thresh} | {nat_us}% | {cert_us}% | {nat_paper}% | {cert_paper}% | {domain} |")
    elif args.hpc or not args.markdown:
        blank_line()
        out_lines.append(f"{'Config':<8} {'Best τ':<7} {'Nat(us)':<9} {'Cert(us)':<9} {'Nat(paper)':<11} {'Cert(paper)':<11} {'Domain':<6}")
        out_lines.append("-" * 62)
        for r in match_rows:
            cid, thresh, nat_us, cert_us, nat_paper, cert_paper, domain = r
            out_lines.append(f"{cid:<8} {thresh:<7} {nat_us:<9} {cert_us:<9} {nat_paper:<11} {cert_paper:<11} {domain:<6}")

    out_lines.append("")

    # ── Output ──
    output_text = "\n".join(out_lines)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text)
            f.write("\n")
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(output_text)


if __name__ == "__main__":
    main()

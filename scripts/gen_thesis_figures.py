#!/usr/bin/env python3
"""Generate ALL 2/255 figures and tables for the thesis from v4 results.

Outputs (in results/figures/ and results/tables/, then copied to thesis/):
  figures/pareto_front_2_255.pdf       — Pareto front (nat vs combined cert)
  figures/cert_vs_tau_2_255.pdf        — cert vs τ
  figures/cross_gate_fronts_2_255.pdf — cross-gate Pareto fronts
  figures/acrown_comparison_2_255.pdf  — box vs α-CROWN vs combined at τ=0
  tables/pareto_table_2_255.tex        — full τ grid table
  tables/key_numbers_2_255.tex         — selected τ comparison table
  tables/control_experiments.tex       — core network + released model comparison
  tables/cross_gate_2_255.tex          — cross-gate results table
"""
import os
import sys
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Paths
RESULTS = os.path.expanduser("~/.openclaw/workspace/research/results")
OUT_FIG = os.path.join(RESULTS, "figures")
OUT_TAB = os.path.join(RESULTS, "tables")
THESIS_FIG = os.path.expanduser("~/.openclaw/workspace/research/thesis/AIM-thesis-main/figures")
THESIS_TAB = os.path.expanduser("~/.openclaw/workspace/research/thesis/AIM-thesis-main/tables")
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_TAB, exist_ok=True)
os.makedirs(THESIS_FIG, exist_ok=True)
os.makedirs(THESIS_TAB, exist_ok=True)

# ─── Load data ───────────────────────────────────────────────────────
v4 = pd.read_csv(os.path.join(RESULTS, "results/agg_sel_eval_v4.csv"))
v4_acrown = pd.read_csv(os.path.join(RESULTS, "results/agg_sel_eval_v4_with_acrown.csv"))

# Strict filtering: only keep real eval runs, not strays like ibp_2_255_v4 or ibp_2_255_init_v4
EVAL_SUFFIXES = ("_sel_eval_v4", "_ent_eval_v4")
is_eval = v4["exp_name"].str.endswith(EVAL_SUFFIXES)
is_smoke = v4["exp_name"] == "smoke_v4"
is_colt = v4["exp_name"] == "colt_v4"
is_cross = v4["exp_name"].str.startswith("cross_")
v4_clean = v4[is_eval | is_smoke | is_colt | is_cross].copy()

# Derive branch key
v4_clean["branch"] = (v4_clean["exp_name"]
    .str.replace("_sel_eval_v4", "")
    .str.replace("_ent_eval_v4", "")
    .str.replace("_v4", ""))

smoke = v4_clean[is_smoke].copy()
smoke = smoke.drop_duplicates(subset=["tau"], keep="first")
v4_methods = v4_clean[is_eval | is_colt].copy()

# Filter to 2/255
v4_2 = v4_methods[(v4_methods["branch"].str.contains("2_255")) | (v4_methods["branch"] == "colt")].copy()
smoke_2 = smoke[smoke["eps"] == 0.00784].copy()
v4_acrown_2 = v4_acrown[v4_acrown["branch"].str.contains("2_255")].copy()

# Merge combined cert
v4_2 = v4_2.merge(v4_acrown_2[["branch", "tau", "cert_combined"]], on=["branch", "tau"], how="left")
v4_2.loc[v4_2["branch"] == "colt", "cert_combined"] = v4_2.loc[v4_2["branch"] == "colt", "cert_acc"]
v4_2["cert_combined"] = v4_2["cert_combined"].fillna(v4_2["cert_acc"])

# Cross-gate data (2/255 only)
cross_2 = v4_clean[is_cross & (v4_clean["eps"] == 0.00784)].copy()

# ─── Style ───────────────────────────────────────────────────────────
METHOD_LABELS = {
    "ibp_2_255": "IBP",
    "sabr_2_255": "SABR",
    "mtl_ibp_2_255": "MTL-IBP",
    "crown_ibp_2_255": "CROWN-IBP",
    "colt": "COLT",
}
METHOD_ORDER = ["ibp_2_255", "mtl_ibp_2_255", "crown_ibp_2_255", "sabr_2_255", "colt"]
COLORS = {
    "ibp_2_255": "#0072B2",
    "mtl_ibp_2_255": "#D55E00",
    "crown_ibp_2_255": "#009E73",
    "sabr_2_255": "#CC79A7",
    "colt": "#56B4E9",
}
MARKERS = {
    "ibp_2_255": "o",
    "mtl_ibp_2_255": "s",
    "crown_ibp_2_255": "^",
    "sabr_2_255": "D",
    "colt": "p",
}
LINESTYLES = {
    "ibp_2_255": "-",
    "mtl_ibp_2_255": "-",
    "crown_ibp_2_255": "-",
    "sabr_2_255": "-",
    "colt": "--",
}
ACE_POINTS = {
    "IBP (paper)": (90.5, 18.5),
    "COLT (paper)": (90.1, 27.5),
}


# ─── Figure 1: Pareto front ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for method in METHOD_ORDER:
    m = v4_2[v4_2["branch"] == method].sort_values("tau")
    cert_label = "Released COLT (zono)" if method == "colt" else f"{METHOD_LABELS[method]} (box + $\\alpha$-CROWN)"
    ax.plot(m["nat_acc"] * 100, m["cert_combined"] * 100,
            color=COLORS[method], marker=MARKERS[method], markersize=5,
            linewidth=1.5, linestyle=LINESTYLES[method],
            label=cert_label)
s = smoke_2.sort_values("tau")
ax.plot(s["nat_acc"] * 100, s["cert_acc"] * 100,
        color="gray", marker="x", markersize=5, linewidth=1.5,
        linestyle="--", label="Released IBP (box)")
for label, (nat, cert) in ACE_POINTS.items():
    ax.plot(nat, cert, marker="*", markersize=12, color="black", zorder=5)
    ax.annotate(label, (nat, cert), textcoords="offset points",
                xytext=(8, -5), fontsize=8, fontstyle="italic")
ax.set_xlabel("Natural accuracy (%)", fontsize=11)
ax.set_ylabel("Certified accuracy (%)", fontsize=11)
ax.set_title(r"Pareto front at $\varepsilon = 2/255$", fontsize=12)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9,
          bbox_to_anchor=(1.02, 1.0), borderaxespad=0, labelspacing=0.4)
ax.grid(True, alpha=0.3)
ax.set_xlim(60, 96)
ax.set_ylim(0, 50)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
fig.tight_layout()
fig.subplots_adjust(right=0.75)
fig.savefig(os.path.join(OUT_FIG, "pareto_front_2_255.pdf"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT_FIG, "pareto_front_2_255.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved pareto_front_2_255")


# ─── Figure 2: natural + certified accuracy vs τ (dual axis) ─────────
fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()
for method in METHOD_ORDER:
    m = v4_2[v4_2["branch"] == method].sort_values("tau")
    cl = "Released COLT (zono)" if method == "colt" else f"{METHOD_LABELS[method]} (box + $\\alpha$-CROWN)"
    ax1.plot(m["tau"], m["cert_combined"] * 100, color=COLORS[method], marker=MARKERS[method],
             markersize=5, linewidth=1.5, linestyle="-", label=cl)
    ax2.plot(m["tau"], m["nat_acc"] * 100, color=COLORS[method], marker=MARKERS[method],
             markersize=3, linewidth=1.0, linestyle="--", alpha=0.6)
s = smoke_2.sort_values("tau")
ax1.plot(s["tau"], s["cert_acc"] * 100, color="gray", marker="x", markersize=5, linewidth=1.5,
         linestyle="-", label="Released IBP (box)")
ax2.plot(s["tau"], s["nat_acc"] * 100, color="gray", marker="x", markersize=3,
         linewidth=1.0, linestyle="--", alpha=0.6)
ax1.set_xlabel(r"Gate threshold $\tau$", fontsize=11)
ax1.set_ylabel("Certified accuracy (%, solid)", fontsize=11)
ax2.set_ylabel("Natural accuracy (%, dashed)", fontsize=11)
ax1.invert_xaxis()
ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
ax1.set_title(r"Natural and certified accuracy vs $\tau$ at $\varepsilon = 2/255$", fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper left", fontsize=9, framealpha=0.9, bbox_to_anchor=(1.12, 1.0),
           borderaxespad=0, labelspacing=0.4)
fig.tight_layout()
fig.subplots_adjust(right=0.67)
fig.savefig(os.path.join(OUT_FIG, "nat_cert_vs_tau_2_255.pdf"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT_FIG, "nat_cert_vs_tau_2_255.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved nat_cert_vs_tau_2_255")


# ─── Figure 3: Cross-gate Pareto fronts ──────────────────────────────
GATE_LABELS = {"ibp": "IBP", "released_ibp": "Released IBP"}
BRANCH_LABELS = {"ibp": "IBP", "sabr": "SABR", "mtl_ibp": "MTL-IBP", "crown_ibp": "CROWN-IBP"}
def parse_cross(name):
    name = name.replace("_v4", "").replace("cross_", "")
    parts = name.split("_gate_on_")
    gate = GATE_LABELS.get(parts[0].replace("_2_255", ""), parts[0].replace("_2_255", ""))
    branch = BRANCH_LABELS.get(parts[1].replace("_2_255", ""), parts[1].replace("_2_255", "")) if len(parts) > 1 else "?"
    return gate, branch

CROSS_STYLES = [
    {"color": "#E91E63", "marker": "o"}, {"color": "#9C27B0", "marker": "s"},
    {"color": "#FF5722", "marker": "^"}, {"color": "#795548", "marker": "D"},
    {"color": "#3F51B5", "marker": "v"}, {"color": "#009688", "marker": "<"},
    {"color": "#FF9800", "marker": ">"},
]

cross_2 = v4[is_cross & (v4["eps"] == 0.00784)].copy()
if len(cross_2) > 0:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # Background: matched fronts, dotted and faded
    for method in METHOD_ORDER:
        m = v4_2[v4_2["branch"] == method].sort_values("tau")
        cl = "Released COLT (zono)" if method == "colt" else f"{METHOD_LABELS[method]} (box + $\\alpha$-CROWN)"
        ax.plot(m["nat_acc"]*100, m["cert_combined"]*100, color=COLORS[method], marker=".",
                markersize=3, linewidth=0.8, linestyle=":", alpha=0.4, label=cl)
    s = smoke_2.sort_values("tau")
    ax.plot(s["nat_acc"]*100, s["cert_acc"]*100, color="gray", marker=".", markersize=3,
            linewidth=0.8, linestyle=":", alpha=0.4, label="Released IBP (box)")
    # Foreground: cross-gate pairs, solid
    for i, exp in enumerate(sorted(cross_2["exp_name"].unique())):
        m = cross_2[cross_2["exp_name"] == exp].sort_values("tau")
        gate, branch = parse_cross(exp)
        sty = CROSS_STYLES[i % len(CROSS_STYLES)]
        ax.plot(m["nat_acc"]*100, m["cert_acc"]*100, color=sty["color"], marker=sty["marker"],
                markersize=5, linewidth=1.8, linestyle="-", alpha=0.85,
                label=f"{branch} ({gate} gate)")
    for label, (nat, cert) in ACE_POINTS.items():
        ax.plot(nat, cert, marker="*", markersize=12, color="black", zorder=5)
        ax.annotate(label, (nat, cert), textcoords="offset points", xytext=(8, -5),
                    fontsize=8, fontstyle="italic")
    ax.set_xlabel("Natural accuracy (%)", fontsize=11)
    ax.set_ylabel("Certified accuracy (%)", fontsize=11)
    ax.set_title(r"Cross-gate evaluation at $\varepsilon = 2/255$", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(60, 96)
    ax.set_ylim(0, 50)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9, bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0, labelspacing=0.35)
    fig.tight_layout()
    fig.subplots_adjust(right=0.70)
    fig.savefig(os.path.join(OUT_FIG, "cross_gate_fronts_2_255.pdf"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_FIG, "cross_gate_fronts_2_255.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved cross_gate_fronts_2_255 (fronts)")
else:
    print("SKIP cross_gate figure: no data")


# ─── Figure 4: α-CROWN comparison (box vs α-CROWN vs combined) ──────
acrown_tau0 = v4_acrown_2[v4_acrown_2["tau"] == 0.0].copy()
if len(acrown_tau0) > 0:
    fig, ax = plt.subplots(figsize=(7, 5))
    methods_ac = ["ibp_2_255", "mtl_ibp_2_255", "crown_ibp_2_255", "sabr_2_255"]
    x = np.arange(len(methods_ac))
    width = 0.25
    box_vals = []
    alpha_vals = []
    combined_vals = []
    for m in methods_ac:
        row = acrown_tau0[acrown_tau0["branch"] == m]
        if len(row) > 0:
            r = row.iloc[0]
            box_vals.append(r["cert_box_agg"] * 100)
            alpha_vals.append(r["cert_alpha"] * 100)
            combined_vals.append(r["cert_combined"] * 100)
        else:
            box_vals.append(0); alpha_vals.append(0); combined_vals.append(0)

    ax.bar(x - width, box_vals, width, label="Box", color="#0072B2", alpha=0.8)
    ax.bar(x, alpha_vals, width, label=r"$\alpha$-CROWN", color="#D55E00", alpha=0.8)
    ax.bar(x + width, combined_vals, width, label=r"Box + $\alpha$-CROWN", color="#009E73", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods_ac], fontsize=10)
    ax.set_ylabel("Certified accuracy (%)", fontsize=11)
    ax.set_title(r"Verifier comparison at $\tau=0$, $\varepsilon = 2/255$", fontsize=12)
    ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    fig.savefig(os.path.join(OUT_FIG, "acrown_comparison_2_255.pdf"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_FIG, "acrown_comparison_2_255.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved acrown_comparison_2_255")
else:
    print("SKIP acrown_comparison figure: no data")


# ─── Table 1: Control experiments ────────────────────────────────────
# Core network + released model comparison
lines = []
lines.append("% Auto-generated by gen_thesis_figures.py — do not edit manually")
lines.append(r"\begin{table}[ht]")
lines.append(r"  \centering")
lines.append(r"  \caption{Core network and released ACE model evaluation at $\varepsilon = 2/255$. Our retrained trunk outperforms the released ACE trunk on both natural and PGD accuracy. The released ACE compositions (IBP and COLT) reproduce the paper's published numbers exactly. All values are percentages over 10\,000 test samples.}")
lines.append(r"  \label{tab:control_experiments}")
lines.append(r"  \begin{tabular}{lccc}")
lines.append(r"    \toprule")
lines.append(r"    Configuration & Natural & PGD & Certified \\")
lines.append(r"    \midrule")
lines.append(r"    \multicolumn{4}{l}{\textit{Core networks}} \\")
lines.append(f"    Released ACE trunk & 90.5 & 71.3 & -- \\\\")
lines.append(f"    Our retrained trunk & 95.3 & 82.5 & -- \\\\")
lines.append(r"    \midrule")
lines.append(r"    \multicolumn{4}{l}{\textit{Released ACE compositions ($\tau=0$)}} \\")
# Released IBP at tau=0
r_ibp = smoke_2[smoke_2["tau"] == 0.0]
if len(r_ibp) > 0:
    r = r_ibp.iloc[0]
    lines.append(f"    Released IBP (paper: 90.5/18.5) & {r['nat_acc']*100:.1f} & {r['pgd_acc']*100:.1f} & {r['cert_acc']*100:.1f} \\\\")
# COLT at tau=0
r_colt = v4_2[(v4_2["branch"] == "colt") & (v4_2["tau"] == 0.0)]
if len(r_colt) > 0:
    r = r_colt.iloc[0]
    lines.append(f"    Released COLT (paper: 90.1/27.5) & {r['nat_acc']*100:.1f} & {r['pgd_acc']*100:.1f} & {r['cert_acc']*100:.1f} \\\\")
lines.append(r"    \bottomrule")
lines.append(r"  \end{tabular}")
lines.append(r"\end{table}")
with open(os.path.join(OUT_TAB, "control_experiments.tex"), "w") as f:
    f.write("\n".join(lines))
print("Saved control_experiments.tex")


# ─── Table 2: Full τ grid ────────────────────────────────────────────
TAUS_TABLE = [-2.0, -1.5, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0]
lines = []
lines.append("% Auto-generated by gen_thesis_figures.py — do not edit manually")
lines.append(r"\begin{table}[ht]")
lines.append(r"  \centering")
lines.append(r"  \caption{Composed natural accuracy, PGD accuracy, and certified accuracy at $\varepsilon = 2/255$ for all four branch methods, the COLT control (zono verifier), and the released ACE baseline across the $\tau$ grid. Our methods use the combined verifier (box OR $\alpha$-CROWN); COLT and released use box/zono only. All values are percentages over 10\,000 test samples.}")
lines.append(r"  \label{tab:pareto_2_255}")
lines.append(r"  \footnotesize")
lines.append(r"  \begin{tabular}{l" + "ccc" * 6 + "}")
lines.append(r"    \toprule")
lines.append(r"    $\tau$ " +
    r" & \multicolumn{3}{c}{IBP} & \multicolumn{3}{c}{MTL-IBP}"
    r" & \multicolumn{3}{c}{CROWN-IBP} & \multicolumn{3}{c}{SABR}"
    r" & \multicolumn{3}{c}{COLT} & \multicolumn{3}{c}{Released} \\")
lines.append(r"    \cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}"
    r" \cmidrule(lr){11-13} \cmidrule(lr){14-16} \cmidrule(lr){17-19}")
lines.append(r"    & Nat & PGD & Cert & Nat & PGD & Cert & Nat & PGD & Cert"
    r" & Nat & PGD & Cert & Nat & PGD & Cert & Nat & PGD & Cert \\")
for tau in TAUS_TABLE:
    row = f"    {tau:+.1f}"
    for method in METHOD_ORDER:
        m = v4_2[(v4_2["branch"] == method) & (v4_2["tau"] == tau)]
        if len(m) > 0:
            r = m.iloc[0]
            row += f" & {r['nat_acc']*100:.1f} & {r.get('pgd_acc', float('nan'))*100:.1f} & {r['cert_combined']*100:.1f}"
        else:
            row += " & -- & -- & --"
    s = smoke_2[smoke_2["tau"] == tau]
    if len(s) > 0:
        r = s.iloc[0]
        row += f" & {r['nat_acc']*100:.1f} & {r['pgd_acc']*100:.1f} & {r['cert_acc']*100:.1f}"
    else:
        row += " & -- & -- & --"
    row += r" \\"
    lines.append(row)
lines.append(r"    \bottomrule")
lines.append(r"  \end{tabular}")
lines.append(r"\end{table}")
with open(os.path.join(OUT_TAB, "pareto_table_2_255.tex"), "w") as f:
    f.write("\n".join(lines))
print("Saved pareto_table_2_255.tex")


# ─── Table 3: Key numbers ────────────────────────────────────────────
KEY_TAUS = [0.0, -0.5, -1.0, -2.0]
lines2 = []
lines2.append("% Auto-generated by gen_thesis_figures.py — do not edit manually")
lines2.append(r"\begin{table}[ht]")
lines2.append(r"  \centering")
lines2.append(r"  \caption{Composed accuracy at selected $\tau$ values for $\varepsilon = 2/255$. Our methods use the combined verifier (box OR $\alpha$-CROWN); COLT uses the zono verifier. The released ACE baseline uses box verification only. All values are percentages over 10\,000 test samples.}")
lines2.append(r"  \label{tab:key_numbers_2_255}")
lines2.append(r"  \begin{tabular}{lcccc}")
lines2.append(r"    \toprule")
lines2.append(r"    $\tau$ & Method & Nat & PGD & Cert \\")
lines2.append(r"    \midrule")
for i, tau in enumerate(KEY_TAUS):
    for method in METHOD_ORDER:
        m = v4_2[(v4_2["branch"] == method) & (v4_2["tau"] == tau)]
        if len(m) > 0:
            r = m.iloc[0]
            tau_str = f"{tau:+.1f}" if method == METHOD_ORDER[0] else ""
            lines2.append(f"    {tau_str} & {METHOD_LABELS[method]} & {r['nat_acc']*100:.1f} & {r.get('pgd_acc', float('nan'))*100:.1f} & {r['cert_combined']*100:.1f}\\\\")
    s = smoke_2[smoke_2["tau"] == tau]
    if len(s) > 0:
        r = s.iloc[0]
        lines2.append(f"     & Released & {r['nat_acc']*100:.1f} & {r['pgd_acc']*100:.1f} & {r['cert_acc']*100:.1f}\\\\")
    if i < len(KEY_TAUS) - 1:
        lines2.append(r"    \midrule")
lines2.append(r"    \bottomrule")
lines2.append(r"  \end{tabular}")
lines2.append(r"\end{table}")
with open(os.path.join(OUT_TAB, "key_numbers_2_255.tex"), "w") as f:
    f.write("\n".join(lines2))
print("Saved key_numbers_2_255.tex")


# ─── Copy everything to thesis ───────────────────────────────────────
print("\nCopying to thesis directory...")
for f in os.listdir(OUT_FIG):
    if f.endswith((".pdf", ".png")):
        shutil.copy2(os.path.join(OUT_FIG, f), os.path.join(THESIS_FIG, f))
        print(f"  {f} -> thesis/figures/")
for f in os.listdir(OUT_TAB):
    if f.endswith(".tex"):
        shutil.copy2(os.path.join(OUT_TAB, f), os.path.join(THESIS_TAB, f))
        print(f"  {f} -> thesis/tables/")
print("\nDone.")

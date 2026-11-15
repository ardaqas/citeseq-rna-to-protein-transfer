"""Build the README figures and summary table from saved result tables only.

Run from the repository root:  .venv/bin/python scripts/make_readme_figures.py
Inputs : results/*.csv (no raw data needed)
Outputs: results/figures/readme_*.png and .svg, results/readme_summary_table.csv
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"

# Colour-blind-checked palette (validated: blue/orange pass all-pair CVD checks on white).
BLUE, ORANGE, GREY, GREY_DARK = "#2a78d6", "#eb6834", "#9a9890", "#52514e"
INK, MUTED, GRID, WHITE = "#0b0b0b", "#6b6a66", "#e6e5df", "#ffffff"

mpl.rcParams.update({
    "figure.facecolor": WHITE, "axes.facecolor": WHITE, "savefig.facecolor": WHITE,
    "savefig.transparent": False, "axes.edgecolor": "#b9b8b0", "axes.labelcolor": INK,
    "axes.titlecolor": INK, "text.color": INK, "xtick.color": GREY_DARK, "ytick.color": GREY_DARK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "legend.fontsize": 10, "figure.titlesize": 14,
    "savefig.dpi": 150, "savefig.bbox": "tight", "svg.fonttype": "none", "svg.hashsalt": "citeseq",
    "font.family": "sans-serif",
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.svg", metadata={"Date": None})  # no timestamp: deterministic SVG
    plt.close(fig)


# ---------------------------------------------------------------- summary table
rep = pd.read_csv(RES / "phase5_report_groups.csv")
pf = pd.read_csv(RES / "phase5_per_fold.csv")
m3 = pd.read_csv(RES / "phase5_m3_site_folds.csv")


def rep_val(group, model, col):
    r = rep[(rep["report group"] == group) & (rep["model"] == model)]
    return float(r[col].iloc[0])


def overall(split, folds=None, subset="all", model="ridge 3a"):
    d = pf[(pf["split"] == split) & (pf["subset"] == subset) & (pf["model"] == model)]
    if folds is not None:
        d = d[d["fold"].isin(folds)]
    return float(d["pearson all cells"].mean())


donors29 = [f"donor{i}" for i in range(2, 10)]
mlp_R = ["MLP s0", "MLP s1", "MLP s2"]
rows = []
# R
rows.append({"split": "R: random cells", "test cells": "13,540 (all 12 samples, seen)",
             "overall r: cell-type mean": overall("R", model="cell-type mean"),
             "overall r: ridge": overall("R"),
             "overall r: MLP": np.mean([overall("R", model=m) for m in mlp_R]),
             "within type x sample r: ridge": rep_val("R test", "ridge 3a", "M1 within type x sample"),
             "within type x sample r: MLP": np.mean([rep_val("R test", m, "M1 within type x sample") for m in mlp_R]),
             "shift captured: ridge": rep_val("R test", "ridge 3a", "M2 shift captured"),
             "shift captured: MLP": np.mean([rep_val("R test", m, "M2 shift captured") for m in mlp_R])})
for label, grp, folds in [("D: new donor, site seen (donors 2-9)", "D donors 2-9", donors29),
                          ("D: donor1 held out (sites seen)", "D donor1", ["donor1"])]:
    rows.append({"split": label, "test cells": "per fold: one donor",
                 "overall r: cell-type mean": overall("D", folds, model="cell-type mean"),
                 "overall r: ridge": overall("D", folds), "overall r: MLP": overall("D", folds, model="MLP s0"),
                 "within type x sample r: ridge": rep_val(grp, "ridge 3a", "M1 within type x sample"),
                 "within type x sample r: MLP": rep_val(grp, "MLP s0", "M1 within type x sample"),
                 "shift captured: ridge": rep_val(grp, "ridge 3a", "M2 shift captured"),
                 "shift captured: MLP": rep_val(grp, "MLP s0", "M2 shift captured")})
for label, grp, sub in [("C: unseen site, donor1 cells (donor seen)", "C donor1 cells", "donor1"),
                        ("C: unseen site, other donors (both unseen)", "C other donors", "other donors")]:
    rows.append({"split": label, "test cells": "per fold: one site",
                 "overall r: cell-type mean": overall("C", subset=sub, model="cell-type mean"),
                 "overall r: ridge": overall("C", subset=sub), "overall r: MLP": overall("C", subset=sub, model="MLP s0"),
                 "within type x sample r: ridge": rep_val(grp, "ridge 3a", "M1 within type x sample"),
                 "within type x sample r: MLP": rep_val(grp, "MLP s0", "M1 within type x sample"),
                 "shift captured: ridge": rep_val(grp, "ridge 3a", "M2 shift captured"),
                 "shift captured: MLP": rep_val(grp, "MLP s0", "M2 shift captured")})
summary = pd.DataFrame(rows)
summary.to_csv(RES / "readme_summary_table.csv", index=False)
site_overall = m3.groupby("model")["pearson all cells"].mean()

# ---------------------------------------------------------------- main figure
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), gridspec_kw={"width_ratios": [1.05, 1.15, 1.0]})
fig.suptitle("Sample-level shifts transfer only to sites seen in training; a modest within-sample signal "
             "transfers everywhere", x=0.01, ha="left", fontweight="bold")

# A: overall accuracy, R vs D vs C
ax = axes[0]
splits = ["random\n(R)", "new donor\n(D, 9 folds)", "unseen site\n(C, 4 folds)"]
vals = {
    "cell-type mean (labels)": [overall("R", model="cell-type mean"), overall("D", model="cell-type mean"), site_overall["cell-type mean"]],
    "ridge": [overall("R"), overall("D"), site_overall["ridge 3a"]],
    "MLP": [np.mean([overall("R", model=m) for m in mlp_R]), overall("D", model="MLP s0"), site_overall["MLP s0"]],
}
x = np.arange(3)
w = 0.26
for (name, v), off, colr in zip(vals.items(), (-w, 0, w), (GREY, BLUE, ORANGE)):
    ax.bar(x + off, v, w * 0.92, color=colr, label=name)
    for xi, vi in zip(x + off, v):
        ax.text(xi, vi + 0.012, f"{vi:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
ax.set_xticks(x, splits)
ax.set_ylim(0, 0.95)
ax.set_ylabel("Pearson r, all test cells\n(median over 110 proteins)")
ax.set_title("A  Overall accuracy: the RNA models' lead\nover cell-type means vanishes on unseen sites", loc="left")
ax.legend(frameon=False, loc="upper right", ncol=1, fontsize=9)
ax.grid(axis="x", visible=False)

# B: shift captured
groups = ["R test", "D donors 2-9", "D donor1", "C donor1 cells", "C other donors"]
glabels = ["random\n(seen)", "new donor\n(site seen)", "donor1\n(sites seen)", "donor1 at\nunseen site",
           "new donor at\nunseen site"]
xg = np.arange(len(groups))
ax = axes[1]
ax.axvspan(2.5, 4.5, color="#f4f3ef", zorder=0)
ax.text(3.5, 0.67, "site unseen", ha="center", fontsize=9, color=MUTED)
for model, colr, mk, off, lab in [("ridge 3a", BLUE, "s", -0.11, "ridge"), ("MLP", ORANGE, "D", 0.11, "MLP")]:
    ys = []
    for g in groups:
        if model == "MLP":
            ys.append(np.mean([rep_val(g, m, "M2 shift captured") for m in mlp_R]) if g == "R test" else rep_val(g, "MLP s0", "M2 shift captured"))
        else:
            ys.append(rep_val(g, model, "M2 shift captured"))
    ax.plot(xg + off, ys, color=colr, lw=1.2, alpha=0.5, zorder=2)
    ax.scatter(xg + off, ys, color=colr, marker=mk, s=60, edgecolors=WHITE, linewidths=1, zorder=3, label=lab)
ax.axhline(0, color=MUTED, lw=1)
ax.set_xticks(xg, glabels, fontsize=9)
ax.set_ylim(-0.05, 0.72)
ax.set_ylabel("share of between-sample shift captured\n(0 = no better than cell-type mean)")
ax.set_title("B  Sample-level shift is captured for seen samples,\nnot for unseen sites", loc="left")
ax.legend(frameon=False, loc="upper center")
ax.grid(axis="x", visible=False)

# C: within type x sample
ax = axes[2]
ax.axvspan(2.5, 4.5, color="#f4f3ef", zorder=0)
for model, colr, mk, off, lab, face in [("own gene", GREY_DARK, "o", -0.16, "own gene only", "none"),
                                       ("ridge 3a", BLUE, "s", 0, "ridge", None), ("MLP", ORANGE, "D", 0.16, "MLP", None)]:
    ys = []
    for g in groups:
        if model == "MLP":
            ys.append(np.mean([rep_val(g, m, "M1 within type x sample") for m in mlp_R]) if g == "R test" else rep_val(g, "MLP s0", "M1 within type x sample"))
        else:
            ys.append(rep_val(g, model, "M1 within type x sample"))
    ax.scatter(xg + off, ys, marker=mk, s=60, zorder=3, label=lab,
               facecolors=colr if face is None else "none", edgecolors=colr if face == "none" else WHITE,
               linewidths=1.4 if face == "none" else 1)
ax.set_ylim(0, 0.16)
ax.set_xticks(xg, glabels, fontsize=9)
ax.set_ylabel("Pearson r within cell type x sample")
ax.set_title("C  Within-sample, cell-level signal is modest\nbut stable across splits", loc="left")
ax.legend(frameon=False, loc="lower right")
ax.grid(axis="x", visible=False)
fig.tight_layout(rect=(0, 0, 1, 0.96))
save(fig, "readme_main_generalization")

# ---------------------------------------------------------------- splits schematic
cells = pd.DataFrame({"donor": ["donor1"] * 4 + [f"donor{i}" for i in range(2, 10)],
                      "site": ["site1", "site2", "site3", "site4", "site1", "site1", "site2", "site2",
                               "site3", "site3", "site4", "site4"]})
donors = [f"donor{i}" for i in range(1, 10)]
sites = ["site1", "site2", "site3", "site4"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
fig.suptitle("What the model sees in training under each split (one example fold shown for D and C)",
             x=0.01, ha="left", fontweight="bold")
titles = ["R  random cells: every sample is split\n70% train / 15% val / 15% test",
          "D  leave one donor out (here donor7):\nits site stays in training via donor1 + donor6",
          "C  leave one site out (here site3): donor1's\nother sites stay in training; donor6/7 unseen"]
for ax, kind, title in zip(axes, ["R", "D", "C"], titles):
    for _, r in cells.iterrows():
        xi, yi = sites.index(r["site"]), donors.index(r["donor"])
        if kind == "R":
            ax.add_patch(Rectangle((xi + 0.06, yi + 0.1), 0.62, 0.8, color=BLUE))
            ax.add_patch(Rectangle((xi + 0.68, yi + 0.1), 0.13, 0.8, color="#b9b8b0"))
            ax.add_patch(Rectangle((xi + 0.81, yi + 0.1), 0.13, 0.8, color=ORANGE))
        else:
            test = (r["donor"] == "donor7") if kind == "D" else (r["site"] == "site3")
            ax.add_patch(Rectangle((xi + 0.06, yi + 0.1), 0.88, 0.8, color=ORANGE if test else BLUE))
            ax.text(xi + 0.5, yi + 0.5, "test" if test else "train", ha="center", va="center",
                    fontsize=9, color=WHITE, fontweight="bold")
    ax.set_xlim(0, 4); ax.set_ylim(9, 0)
    ax.set_xticks(np.arange(4) + 0.5, sites)
    ax.set_yticks(np.arange(9) + 0.5, donors)
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11)
handles = [Rectangle((0, 0), 1, 1, color=BLUE), Rectangle((0, 0), 1, 1, color="#b9b8b0"),
           Rectangle((0, 0), 1, 1, color=ORANGE)]
fig.legend(handles, ["training cells", "validation cells (R only)", "test cells"], loc="lower center",
           ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.text(0.01, -0.02, "Blank: donor not measured at that site.", fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.05, 1, 0.95))
save(fig, "readme_splits")

# ---------------------------------------------------------------- own gene figure
p2 = pd.read_csv(RES / "phase2_protein_rna_correlation.csv")
p2 = p2[p2["analysis_set"] == "primary"]
fig, ax = plt.subplots(figsize=(7.2, 5.4))
ax.plot([-0.1, 1], [-0.1, 1], color=MUTED, ls="--", lw=1)
ax.text(0.62, 0.66, "equal", color=MUTED, fontsize=9, rotation=36)
ax.scatter(p2["pearson_all"], p2["pearson_within_w"], s=34, color=BLUE, edgecolors=WHITE, linewidths=0.7, zorder=3)
own_off = {"IgD": (5, 4), "CD8": (-12, 10), "CD127": (8, 2), "CD3": (8, -2), "CD20": (-38, -12)}
for _, r in p2.nlargest(5, "pearson_all").iterrows():
    ax.annotate(r["protein"], (r["pearson_all"], r["pearson_within_w"]), xytext=own_off.get(r["protein"], (4, 4)),
                textcoords="offset points", fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=0, shrinkB=3))
ax.set_xlim(-0.12, 1); ax.set_ylim(-0.12, 1)
ax.set_xlabel("Pearson r: protein vs. own-gene RNA, all cells")
ax.set_ylabel("Pearson r within cell type (cell-weighted)")
ax.set_title("A protein's own gene tracks cell type, not variation within a type\n"
             f"(110 proteins; median {p2['pearson_all'].median():.2f} across all cells vs "
             f"{p2['pearson_within_w'].median():.2f} within cell types)", loc="left", fontsize=11.5)
fig.tight_layout()
save(fig, "readme_own_gene")

# ---------------------------------------------------------------- per-protein figure
pp = pd.read_csv(RES / "phase6_per_protein.csv")
pp = pp[pp["analysis_set"] == "primary"]
fig, ax = plt.subplots(figsize=(8.6, 5.8))
scat = ax.scatter(pp["R_overall_ridge"], pp["C_within_type_sample_ridge"], c=pp["eta2_cell_type"], cmap="Blues",
                  vmin=0, vmax=1, s=42, edgecolors=GREY_DARK, linewidths=0.4, zorder=3)
cb = fig.colorbar(scat, ax=ax, pad=0.02)
cb.set_label("share of variance explained by cell type")
label = pd.concat([pp.nlargest(5, "C_within_type_sample_ridge"), pp.nsmallest(4, "R_overall_ridge"),
                   pp.nlargest(3, "R_overall_ridge")]).drop_duplicates("protein")
offsets = {"CD36": (-32, -3), "CD71": (5, 4), "CD134": (-14, -14), "CD163": (5, 5)}
for _, r in label.iterrows():
    ax.annotate(r["protein"], (r["R_overall_ridge"], r["C_within_type_sample_ridge"]), fontsize=9,
                xytext=offsets.get(r["protein"], (5, 3)), textcoords="offset points")
ax.set_xlabel("overall Pearson r, random-split test cells (ridge)")
ax.set_ylabel("within cell type x sample Pearson r,\nunseen sites (ridge)")
ax.set_title("Proteins that differ most between cell types are predicted best overall;\n"
             "CD38, ICAM1 and HLA-DR carry the most transferable within-sample signal", loc="left", fontsize=11.5)
fig.tight_layout()
save(fig, "readme_proteins")

for f in sorted(FIG.glob("readme_*")):
    print(f"{f.name:<40} {f.stat().st_size / 1024:7.0f} KB")
print(summary.round(3).to_string(index=False))

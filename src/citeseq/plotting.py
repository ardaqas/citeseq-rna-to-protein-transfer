"""Small, consistent matplotlib helpers.

Colour rules used throughout the project:
- one highlight colour (blue) against grey context instead of many hues;
- groups with more than 3 members are shown as small multiples (one panel per
  group), never as a many-colour scatter.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

BLUE = "#2a78d6"        # highlight / single series
ORANGE = "#eb6834"      # second series, when strictly needed
CONTEXT = "#c3c2b7"     # grey context points
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def set_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": CONTEXT, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.family": "sans-serif",
    })


def highlight_grid(coords, labels, groups=None, ncols=4, panel_size=2.6, s=0.5,
                   suptitle=None, seed=0):
    """One panel per group: that group in blue on top of all cells in grey.

    coords : (n, 2) embedding; labels : (n,) group label per cell.
    """
    labels = np.asarray(labels).astype(str)
    groups = list(groups) if groups is not None else sorted(set(labels))
    nrows = int(np.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size * ncols, panel_size * nrows),
                             squeeze=False)
    # Shuffle drawing order so no subgroup is systematically hidden underneath.
    order = np.random.default_rng(seed).permutation(len(labels))
    xy, lab = coords[order], labels[order]
    for ax, g in zip(axes.flat, groups):
        sel = lab == g
        ax.scatter(xy[~sel, 0], xy[~sel, 1], s=s, c=CONTEXT, alpha=0.35, linewidths=0, rasterized=True)
        ax.scatter(xy[sel, 0], xy[sel, 1], s=s, c=BLUE, linewidths=0, rasterized=True)
        ax.set_title(f"{g}  (n={sel.sum():,})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    for ax in axes.flat[len(groups):]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, color=INK)
    fig.tight_layout()
    return fig


def boxplot_by_group(ax, values, groups, order=None, log=False, ylabel=None):
    """Single-hue box plot of `values` split by `groups` (categories on the x axis)."""
    values, groups = np.asarray(values), np.asarray(groups).astype(str)
    order = order if order is not None else sorted(set(groups))
    data = [values[groups == g] for g in order]
    ax.boxplot(data, showfliers=False, widths=0.6, patch_artist=True,
               boxprops=dict(facecolor="#cde2fb", edgecolor=BLUE, linewidth=1),
               medianprops=dict(color=BLUE, linewidth=2),
               whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
    ax.set_xticks(range(1, len(order) + 1), order, rotation=45, ha="right")
    if log:
        ax.set_yscale("log")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    return ax

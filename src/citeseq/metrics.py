"""Per-protein evaluation metrics used from Phase 3 on.

For each protein:
* pearson_all      Pearson r over all evaluated cells
* within_w / within_unw / within_median
                   Pearson r inside each cell type with >= min_cells evaluated cells,
                   summarized with cell-count weights / without weights / median
* n_types, n_types_nan
                   cell types used, and cell types where r is undefined
* rmse

Undefined values are NaN and are never replaced by 0. r is undefined when the
prediction or the truth is constant: everywhere for a global-mean prediction, and
inside every cell type for a cell-type-mean prediction.
"""
import numpy as np
import pandas as pd

from .correlation import pearson_columns, summarize_within, within_group_pearson


def evaluate(y_true, y_pred, cell_type, names, min_cells=50):
    """Return a per-protein DataFrame (index = protein names)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError("shape mismatch")
    out = pd.DataFrame(index=pd.Index(names, name="protein"))
    out["pearson_all"] = pearson_columns(y_pred, y_true)
    w = summarize_within(within_group_pearson(y_pred, y_true, cell_type, min_cells), y_true.shape[1])
    out["within_w"] = w["weighted_mean"].to_numpy()
    out["within_unw"] = w["mean"].to_numpy()
    out["within_median"] = w["median"].to_numpy()
    out["n_types"] = (w["n_groups"] + w["n_groups_nan"]).to_numpy()
    out["n_types_nan"] = w["n_groups_nan"].to_numpy()
    out["rmse"] = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))
    return out


def headline(per_protein, proteins, metrics=("pearson_all", "within_w", "within_unw", "rmse")):
    """Median and IQR across `proteins` for each metric, plus how many are defined.

    A metric that is undefined for every protein stays NaN ("undefined").
    """
    d = per_protein.loc[list(proteins)]
    rows = {}
    for m in metrics:
        v = d[m]
        rows[m] = {"median": v.median(), "q25": v.quantile(0.25), "q75": v.quantile(0.75),
                   "n_defined": int(v.notna().sum()), "n_proteins": len(v)}
    return pd.DataFrame(rows).T

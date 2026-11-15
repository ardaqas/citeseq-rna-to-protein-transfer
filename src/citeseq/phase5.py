"""Phase 5: refit frozen models per fold and the bias / shift metrics.

Every learned step is fit on the fold's training cells only. Test cells enter only
through `transform`/`predict`; their protein values are never passed to any fit.
"""
import numpy as np
import pandas as pd

from .baselines import CellTypeMean, HVGPCAFeatures, OwnGeneLinear, RidgePath
from .mlp import TargetScaler, predict as mlp_predict, train_mlp
from .splits import inner_split


def fit_predict_fold(X, Y, R, obs, gene_names, train, test, cfg3, cfg4, mlp_seeds=(0,),
                     inner_frac=0.15, mlp_val=None, max_epochs=200, patience=15, checkpoint_fn=None):
    """Fit all models on `train` cells, predict `test` cells.

    MLP early stopping: if `mlp_val` (boolean mask, disjoint from train and test) is
    given, it is used as the stopping set and the MLP trains on all `train` cells
    (Phase 4 setup for the random split). Otherwise an inner split of the training
    cells is used (`inner_frac`, stratified by cell type).
    One MLP per seed in `mlp_seeds` (keys "MLP s<seed>"); `checkpoint_fn(seed)` gives
    its checkpoint path (or None).
    Returns (predictions dict, state dict).
    """
    train, test = np.asarray(train, bool), np.asarray(test, bool)
    assert not (train & test).any()
    ct = obs["cell_type"].astype(str).to_numpy()
    feats = HVGPCAFeatures(n_top_genes=cfg3["features"]["hvg"], max_pcs=None, seed=cfg3["seed"]).fit(
        X[train], obs.loc[train], gene_names)
    n_pcs, alpha = cfg3["3a_ridge_pca"]["n_pcs"], cfg3["3a_ridge_pca"]["alpha"]
    pred = {}
    pred["cell-type mean"] = CellTypeMean().fit(Y[train], ct[train]).predict(ct[test])
    pred["own gene"] = OwnGeneLinear().fit(R[train], Y[train]).predict(R[test])
    pred["ridge 3a"] = RidgePath().fit(feats.transform_pca(X[train], n_pcs), Y[train], [alpha]).predict(
        feats.transform_pca(X[test], n_pcs), alpha)

    if mlp_val is None:
        tr_idx = np.flatnonzero(train)
        m_in, m_va = inner_split(ct[train], frac=inner_frac, seed=0)
        fit_idx, stop_idx = tr_idx[m_in], tr_idx[m_va]
    else:
        mlp_val = np.asarray(mlp_val, bool)
        assert not (mlp_val & (train | test)).any()
        fit_idx, stop_idx = np.flatnonzero(train), np.flatnonzero(mlp_val)
    ys = TargetScaler().fit(Y[fit_idx])          # target scaling: MLP training cells only
    H_fit, H_stop = feats.transform_hvg(X[fit_idx]), feats.transform_hvg(X[stop_idx])
    b = cfg4["best"]
    H_test = feats.transform_hvg(X[test])
    mlp_info = {}
    for seed in mlp_seeds:
        out = train_mlp(H_fit, ys.transform(Y[fit_idx]), H_stop, ys.transform(Y[stop_idx]), hidden=b["hidden"],
                        dropout=b["dropout"], weight_decay=b["weight_decay"], seed=seed, lr=b["lr"],
                        batch_size=b["batch_size"], max_epochs=max_epochs, patience=patience,
                        checkpoint=None if checkpoint_fn is None else checkpoint_fn(seed))
        pred[f"MLP s{seed}"] = mlp_predict(out["model"], H_test, ys)
        mlp_info[seed] = dict(best_epoch_0based=out["best_epoch"], epochs_run=len(out["history"]["val_loss"]),
                              loaded=out.get("loaded", False))
    state = dict(hvg_mask=feats.hvg_mask_.copy(), x_scaler_mean=feats.scaler_.mean_.copy(),
                 y_scaler_mean=ys.mean_.copy(), y_scaler_sd=ys.sd_.copy(), n_mlp_fit=len(fit_idx),
                 n_mlp_stop=len(stop_idx), mlp=mlp_info, sd_train=Y[train].std(axis=0),
                 train_types=sorted(set(ct[train])))
    return pred, state


def group_bias(y_true, y_pred, groups, sd, min_cells, keep=None):
    """Standardized bias per (group, protein): (mean pred - mean truth) / sd.

    Only groups with >= min_cells cells (and, if given, cells with keep=True).
    Returns a long DataFrame: group, j, n, b.
    """
    groups = np.asarray(groups).astype(str)
    keep = np.ones(len(groups), bool) if keep is None else np.asarray(keep, bool)
    rows = []
    for g in pd.unique(groups[keep]):
        sel = keep & (groups == g)
        n = int(sel.sum())
        if n < min_cells:
            continue
        b = (y_pred[sel].mean(axis=0) - y_true[sel].mean(axis=0)) / sd
        rows.append(pd.DataFrame({"group": g, "j": np.arange(y_true.shape[1]), "n": n, "b": b}))
    if not rows:
        return pd.DataFrame(columns=["group", "j", "n", "b"])
    return pd.concat(rows, ignore_index=True)


def weighted_abs_bias(long, cols=None):
    d = long if cols is None else long[long["j"].isin(cols)]
    return float((d["n"] * d["b"].abs()).sum() / d["n"].sum()) if len(d) else np.nan


def shift_captured(long_model, long_ref, cols=None):
    """1 - sum(n|b_model|) / sum(n|b_ref|) over the same (group, protein) pairs (ratio of sums)."""
    m = long_model.merge(long_ref, on=["group", "j", "n"], suffixes=("_m", "_r"))
    if cols is not None:
        m = m[m["j"].isin(cols)]
    den = (m["n"] * m["b_r"].abs()).sum()
    return float(1 - (m["n"] * m["b_m"].abs()).sum() / den) if den > 0 else np.nan

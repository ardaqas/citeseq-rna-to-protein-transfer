"""Shared input preparation for modelling phases (3, 4, ...).

Loads RNA counts and protein CLR, applies the saved split, and returns only the
training and validation parts. Test cells are never returned, so code that uses
this function cannot touch them by accident.
"""
import h5py
import numpy as np
import scanpy as sc

from .correlation import load_protein_gene_map, mapped_log_expression
from .io import load_cite, _read_dataframe
from .paths import CITE_H5AD, REPO_ROOT
from .splits import load_split, make_random_split


def load_train_val(split_path, subset=None, seed=0):
    """Return a dict with train/val inputs and targets.

    Keys: X_tr, X_va (log1p CP10k, sparse), Y_tr, Y_va (protein CLR, dense),
    R_tr, R_va (own-gene RNA, NaN for proteins without a measured gene),
    obs_tr, obs_va, proteins, pmap, gene_names, n_test.
    With `subset`, a random subset of cells is used and a fresh split is made
    (for quick tests only).
    """
    with h5py.File(CITE_H5AD) as f:
        var = _read_dataframe(f["var"])
    mask = None
    if subset:
        mask = np.zeros(90_261, bool)
        mask[np.random.default_rng(seed).choice(mask.size, subset, replace=False)] = True
    rna = load_cite(CITE_H5AD, "layers/counts", cell_mask=mask, var_mask=(var["feature_types"] == "GEX").to_numpy())
    adt = load_cite(CITE_H5AD, "X", cell_mask=mask, var_mask=(var["feature_types"] == "ADT").to_numpy())
    proteins = list(adt.var_names)
    pmap = load_protein_gene_map(REPO_ROOT / "resources/protein_gene_map.csv", rna.var_names)
    pmap = pmap.set_index("protein").loc[proteins]

    split = make_random_split(rna.obs, seed=seed) if subset else load_split(split_path, rna.obs_names)

    total = np.asarray(rna.X.sum(axis=1)).ravel()
    usable = pmap["usable"].to_numpy()
    R = np.full((rna.n_obs, len(proteins)), np.nan, dtype=np.float32)
    R[:, usable] = mapped_log_expression(rna.X, rna.var_names, total, pmap.loc[usable, "gene_list"].tolist())
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)

    tr = (split == "train").to_numpy()
    va = (split == "val").to_numpy()
    Y = adt.X.toarray()
    return dict(
        X_tr=rna.X[tr], X_va=rna.X[va], Y_tr=Y[tr], Y_va=Y[va], R_tr=R[tr], R_va=R[va],
        obs_tr=rna.obs.loc[tr].copy(), obs_va=rna.obs.loc[va].copy(),
        proteins=proteins, pmap=pmap, gene_names=np.asarray(rna.var_names),
        n_test=int((split == "test").sum()),
    )


def load_all(cell_mask=None):
    """All cells (optionally a subset): log1p CP10k RNA, protein CLR, own-gene RNA, obs.

    Only for the final evaluation (Phase 5), where held-out donors/sites and the
    random-split test part are needed.
    """
    with h5py.File(CITE_H5AD) as f:
        var = _read_dataframe(f["var"])
    rna = load_cite(CITE_H5AD, "layers/counts", cell_mask=cell_mask, var_mask=(var["feature_types"] == "GEX").to_numpy())
    adt = load_cite(CITE_H5AD, "X", cell_mask=cell_mask, var_mask=(var["feature_types"] == "ADT").to_numpy())
    proteins = list(adt.var_names)
    pmap = load_protein_gene_map(REPO_ROOT / "resources/protein_gene_map.csv", rna.var_names).set_index("protein").loc[proteins]
    total = np.asarray(rna.X.sum(axis=1)).ravel()
    usable = pmap["usable"].to_numpy()
    R = np.full((rna.n_obs, len(proteins)), np.nan, dtype=np.float32)
    R[:, usable] = mapped_log_expression(rna.X, rna.var_names, total, pmap.loc[usable, "gene_list"].tolist())
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    return dict(X=rna.X, Y=adt.X.toarray(), R=R, obs=rna.obs.copy(), proteins=proteins, pmap=pmap,
                gene_names=np.asarray(rna.var_names))

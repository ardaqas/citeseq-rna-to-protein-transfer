"""Streaming statistics for the large sparse matrices of an .h5ad file.

The CSR matrices are read from disk with h5py in blocks of `chunk_rows` cells,
so the full matrix is never held in memory. Statistics are kept separately for
each feature type (GEX = RNA, ADT = protein).
"""
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _decode(values):
    return np.array([v.decode() if isinstance(v, bytes) else v for v in values], dtype=object)


def read_df_column(group, col):
    """Read one column of an anndata dataframe group (obs/var).

    Handles both the current categorical encoding and the legacy one
    (integer codes + names under '__categories/<col>').
    """
    node = group[col]
    if isinstance(node, h5py.Group) and "categories" in node:
        return pd.Categorical.from_codes(node["codes"][:], _decode(node["categories"][:]))
    if "__categories" in group and col in group["__categories"]:
        return pd.Categorical.from_codes(node[:], _decode(group["__categories"][col][:]))
    values = node[:]
    return _decode(values) if values.dtype.kind in "SO" else values


def read_df_index(group):
    return _decode(group[group.attrs.get("_index", "_index")][:])


def iter_csr_chunks(h5file, key, chunk_rows=5000):
    """Yield (row_start, row_stop, csr_block) for a CSR matrix stored at `key`."""
    g = h5file[key]
    if g.attrs.get("encoding-type") not in ("csr_matrix", b"csr_matrix"):
        raise ValueError(f"{key} is not stored as csr_matrix")
    n_rows, n_cols = (int(s) for s in g.attrs["shape"])
    indptr = g["indptr"][:]
    for r0 in range(0, n_rows, chunk_rows):
        r1 = min(r0 + chunk_rows, n_rows)
        a, b = indptr[r0], indptr[r1]
        block = sp.csr_matrix(
            (g["data"][a:b], g["indices"][a:b], indptr[r0 : r1 + 1] - a),
            shape=(r1 - r0, n_cols),
        )
        yield r0, r1, block


def scan_by_feature_type(path, key, feature_types, dense_types=(), chunk_rows=5000):
    """Collect value statistics of matrix `key`, separately per feature type.

    Parameters
    ----------
    path : path to .h5ad
    key : "X" or "layers/<name>"
    feature_types : array (n_vars,) with a label per column, e.g. "GEX"/"ADT"
    dense_types : feature types whose full block is also returned as a dense
        array (only sensible for small blocks such as the 134 ADT columns)

    Returns
    -------
    summary : DataFrame, one row per feature type
    per_cell : dict {ftype: DataFrame with per-cell sum, nnz, min, max}
    dense : dict {ftype: ndarray (n_cells, n_features_of_type)}
    """
    feature_types = np.asarray(feature_types)
    types = list(pd.unique(feature_types))
    cols = {t: np.flatnonzero(feature_types == t) for t in types}
    acc = {t: dict(n_stored=0, n_negative=0, n_non_integer=0, n_explicit_zero=0,
                   min=np.inf, max=-np.inf) for t in types}
    per_cell = {t: {"sum": [], "nnz": [], "min_stored": [], "max_stored": []} for t in types}
    dense = {t: [] for t in dense_types}

    with h5py.File(path, "r") as f:
        n_cells = int(f[key].attrs["shape"][0])
        for _, _, block in iter_csr_chunks(f, key, chunk_rows):
            for t in types:
                sub = block[:, cols[t]].tocsr()
                d = sub.data
                a = acc[t]
                a["n_stored"] += d.size
                a["n_negative"] += int((d < 0).sum())
                a["n_non_integer"] += int((d != np.round(d)).sum())
                a["n_explicit_zero"] += int((d == 0).sum())
                if d.size:
                    a["min"] = min(a["min"], float(d.min()))
                    a["max"] = max(a["max"], float(d.max()))
                pc = per_cell[t]
                pc["sum"].append(np.asarray(sub.sum(axis=1)).ravel())
                pc["nnz"].append(np.diff(sub.indptr))
                # min/max over stored values per cell (NaN for empty rows)
                mins = np.full(sub.shape[0], np.nan)
                maxs = np.full(sub.shape[0], np.nan)
                nz = np.diff(sub.indptr) > 0
                if nz.any():
                    mins[nz] = np.minimum.reduceat(d, sub.indptr[:-1][nz])
                    maxs[nz] = np.maximum.reduceat(d, sub.indptr[:-1][nz])
                pc["min_stored"].append(mins)
                pc["max_stored"].append(maxs)
                if t in dense:
                    dense[t].append(sub.toarray())

    rows = []
    for t in types:
        a = acc[t]
        total = n_cells * cols[t].size
        rows.append({
            "feature_type": t, "n_features": cols[t].size, **a,
            "density": a["n_stored"] / total,
            "frac_non_integer": a["n_non_integer"] / max(a["n_stored"], 1),
        })
    summary = pd.DataFrame(rows).set_index("feature_type")
    per_cell = {t: pd.DataFrame({k: np.concatenate(v) for k, v in pc.items()})
                for t, pc in per_cell.items()}
    dense = {t: np.vstack(v) for t, v in dense.items()}
    return summary, per_cell, dense


def per_cell_ratio(path, key_num, key_den, columns, chunk_rows=5000):
    """Per-cell summary of the element-wise ratio num/den on the given columns.

    Used to test whether one matrix is a per-cell rescaling of another
    (e.g. X = counts / size_factor). If so, within each cell the ratio is
    constant: ratio_min == ratio_max.

    Both matrices must share the same sparsity pattern on `columns`; this is
    checked and reported per cell (`same_pattern`).
    """
    columns = np.asarray(columns)
    out = {"ratio_median": [], "ratio_min": [], "ratio_max": [], "same_pattern": []}
    with h5py.File(path, "r") as f:
        for (_, _, num), (_, _, den) in zip(iter_csr_chunks(f, key_num, chunk_rows),
                                            iter_csr_chunks(f, key_den, chunk_rows)):
            num = num[:, columns].tocsr()
            den = den[:, columns].tocsr()
            num.sort_indices()
            den.sort_indices()
            for i in range(num.shape[0]):
                a0, a1 = num.indptr[i], num.indptr[i + 1]
                b0, b1 = den.indptr[i], den.indptr[i + 1]
                same = (a1 - a0 == b1 - b0) and np.array_equal(num.indices[a0:a1], den.indices[b0:b1])
                out["same_pattern"].append(same)
                if same and a1 > a0:
                    r = num.data[a0:a1] / den.data[b0:b1]
                    out["ratio_median"].append(np.median(r))
                    out["ratio_min"].append(r.min())
                    out["ratio_max"].append(r.max())
                else:
                    for k in ("ratio_median", "ratio_min", "ratio_max"):
                        out[k].append(np.nan)
    return pd.DataFrame(out)

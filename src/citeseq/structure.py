"""Memory-light inspection of an .h5ad file.

Reads only metadata (shapes, dtypes, column names, category counts) with h5py,
so the big expression matrices are never loaded. Used in Phase 0 to verify the
real structure of the file instead of assuming column names.
"""
from pathlib import Path

import h5py
import numpy as np


def _attr(node, key, default=None):
    """Read an HDF5 attribute and decode bytes to str."""
    val = node.attrs.get(key, default)
    if isinstance(val, bytes):
        return val.decode()
    return val


def _matrix_summary(node, n_check=1_000_000):
    """Describe a dense dataset or a sparse (csr/csc) group without loading it."""
    if isinstance(node, h5py.Dataset):
        return f"dense {node.shape} {node.dtype}"
    enc = _attr(node, "encoding-type", "?")
    if "data" not in node:  # not a sparse matrix, e.g. a dataframe stored in obsm
        cols = [c.decode() if isinstance(c, bytes) else c for c in node.attrs.get("column-order", [])]
        return f"{enc} with columns {cols}"
    shape = tuple(int(s) for s in node.attrs.get("shape", ()))
    data = node["data"]
    nnz = data.shape[0]
    density = nnz / (shape[0] * shape[1]) if shape and shape[0] * shape[1] else float("nan")
    # Are the stored values whole numbers? (raw counts are integers;
    # normalized/log values are not). Only the first n_check values are read.
    head = data[: min(n_check, nnz)]
    is_int = bool(np.all(np.equal(np.mod(head, 1), 0))) if head.size else None
    return (
        f"{enc} shape={shape} dtype={data.dtype} nnz={nnz:,} "
        f"density={density:.4f} first_{head.size:,}_values_integer={is_int} "
        f"min={head.min() if head.size else None} max={head.max() if head.size else None}"
    )


def _dataframe_summary(group, max_categories):
    """Print every column of an anndata-encoded dataframe (obs or var)."""
    index_key = _attr(group, "_index", "_index")
    n_rows = group[index_key].shape[0]
    print(f"  rows: {n_rows:,}  (index column: '{index_key}')")
    index_head = [x.decode() if isinstance(x, bytes) else x for x in group[index_key][:3]]
    print(f"  first index values: {index_head}")
    # Older anndata (< 0.8) stored categoricals as integer codes in the column
    # and the category names separately under "__categories/<column>".
    legacy_cats = group["__categories"] if "__categories" in group else {}
    for col in group.attrs.get("column-order", []):
        col = col.decode() if isinstance(col, bytes) else col
        node = group[col]
        codes = cats = None
        if isinstance(node, h5py.Group) and "categories" in node:  # new format
            cats, codes = node["categories"][:], node["codes"][:]
        elif col in legacy_cats:  # legacy format
            cats, codes = legacy_cats[col][:], node[:]
        if cats is not None:
            cats = [c.decode() if isinstance(c, bytes) else c for c in cats]
            n_missing = int((codes < 0).sum())  # code -1 means NaN
            counts = np.bincount(codes[codes >= 0], minlength=len(cats))
            print(f"  - {col}: categorical, {len(cats)} categories, {n_missing:,} missing")
            order = np.argsort(-counts)
            for i in order[:max_categories]:
                print(f"        {cats[i]!s:<35} {counts[i]:>7,}")
            if len(cats) > max_categories:
                print(f"        ... ({len(cats) - max_categories} more)")
        elif isinstance(node, h5py.Dataset):
            vals = node[:]
            if vals.dtype.kind in "biuf":
                n_nan = int(np.isnan(vals).sum()) if vals.dtype.kind == "f" else 0
                print(f"  - {col}: {vals.dtype}, min={np.nanmin(vals)}, max={np.nanmax(vals)}, "
                      f"n_unique={len(np.unique(vals))}, n_nan={n_nan:,}")
            else:
                uniq = np.unique(vals)
                print(f"  - {col}: {vals.dtype}, n_unique={len(uniq)}, e.g. {list(uniq[:3])}")
        else:
            print(f"  - {col}: {_attr(node, 'encoding-type', type(node).__name__)}")


def describe_h5ad(path, max_categories=15):
    """Print the structure of an .h5ad file: X, obs, var, layers, obsm, uns."""
    path = Path(path)
    print(f"File: {path.name}  ({path.stat().st_size / 1e9:.2f} GB on disk)\n")
    with h5py.File(path, "r") as f:
        print("Top-level keys:", list(f.keys()))
        print("\n[X]\n ", _matrix_summary(f["X"]))
        for name in ("obs", "var"):
            print(f"\n[{name}]")
            _dataframe_summary(f[name], max_categories)
        for name in ("layers", "obsm", "varm", "obsp"):
            if name in f and len(f[name]):
                print(f"\n[{name}]")
                for key in f[name]:
                    print(f"  - {key}: {_matrix_summary(f[name][key])}")
        if "uns" in f:
            print("\n[uns] keys:", list(f["uns"].keys()))

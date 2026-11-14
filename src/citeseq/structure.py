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
    for col in _attr(group, "column-order", []) if len(group.attrs.get("column-order", [])) else []:
        col = col.decode() if isinstance(col, bytes) else col
        node = group[col]
        if isinstance(node, h5py.Group) and "categories" in node:  # categorical
            cats = [c.decode() if isinstance(c, bytes) else c for c in node["categories"][:]]
            counts = np.bincount(node["codes"][:].clip(min=0), minlength=len(cats))
            print(f"  - {col}: categorical, {len(cats)} categories")
            order = np.argsort(-counts)
            for i in order[:max_categories]:
                print(f"        {cats[i]!s:<35} {counts[i]:>7,}")
            if len(cats) > max_categories:
                print(f"        ... ({len(cats) - max_categories} more)")
        elif isinstance(node, h5py.Dataset):
            vals = node[:]
            if vals.dtype.kind in "biuf":
                print(f"  - {col}: {vals.dtype}, min={vals.min()}, max={vals.max()}, "
                      f"n_unique={len(np.unique(vals))}")
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

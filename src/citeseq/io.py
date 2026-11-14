"""Load the CITE-seq .h5ad into AnnData with a chosen matrix as X.

The file stores two full matrices (X and layers['counts']). Loading only the one
we need halves memory use. obs/var are decoded with `read_df_column`, which
understands the file's legacy categorical encoding.
"""
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

from .matrix_stats import iter_csr_chunks, read_df_column, read_df_index


def _read_dataframe(group):
    cols = [c.decode() if isinstance(c, bytes) else c for c in group.attrs.get("column-order", [])]
    df = pd.DataFrame({c: read_df_column(group, c) for c in cols})
    df.index = read_df_index(group)
    return df


def load_cite(path, matrix="layers/counts", cell_mask=None, chunk_rows=5000):
    """Return AnnData with `matrix` as X, plus obs, var and dense obsm entries.

    Parameters
    ----------
    matrix : "layers/counts" (raw counts) or "X" (provided normalized values)
    cell_mask : optional boolean array (n_cells,) to load only some cells.
        The file is still streamed in chunks, so memory scales with the subset.
    """
    with h5py.File(path, "r") as f:
        obs = _read_dataframe(f["obs"])
        var = _read_dataframe(f["var"])
        if cell_mask is None:
            cell_mask = np.ones(len(obs), dtype=bool)
        cell_mask = np.asarray(cell_mask, dtype=bool)

        blocks = [block[cell_mask[r0:r1]] for r0, r1, block in iter_csr_chunks(f, matrix, chunk_rows)]
        X = sp.vstack(blocks, format="csr")

        obsm = {}
        for key, node in f["obsm"].items():
            if isinstance(node, h5py.Dataset):
                obsm[key] = node[:][cell_mask]
            else:  # dataframe stored in obsm (isotype controls)
                obsm[key] = _read_dataframe(node).loc[obs.index[cell_mask]]

    adata = ad.AnnData(X=X, obs=obs.loc[cell_mask].copy(), var=var, obsm=obsm)
    adata.uns["source_matrix"] = matrix
    return adata


def split_modalities(adata, key="feature_types"):
    """Split a joint AnnData into (rna, protein) views copied to new objects."""
    rna = adata[:, adata.var[key] == "GEX"].copy()
    adt = adata[:, adata.var[key] == "ADT"].copy()
    return rna, adt


def csr_nbytes(m):
    """Memory used by a scipy CSR matrix, in bytes."""
    return m.data.nbytes + m.indices.nbytes + m.indptr.nbytes

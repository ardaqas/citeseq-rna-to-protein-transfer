"""Protein vs. same-gene RNA correlation, overall and within groups (Phase 2)."""
import numpy as np
import pandas as pd
import scipy.sparse as sp


def load_protein_gene_map(path, available_genes):
    """Read the hand-made protein -> gene(s) table and flag missing genes.

    `genes` may list several genes separated by ';' (their counts are summed).
    A protein is usable only if all of its genes are present in the data.
    """
    m = pd.read_csv(path)
    available = set(available_genes)
    m["gene_list"] = m["genes"].str.split(";")
    m["missing_genes"] = m["gene_list"].apply(lambda gs: [g for g in gs if g not in available])
    m["usable"] = m["missing_genes"].str.len() == 0
    return m


def mapped_log_expression(counts, gene_names, total_counts, gene_lists, target_sum=1e4):
    """log1p(CP10k) of the summed raw counts of each gene list.

    counts : (cells x genes) sparse raw counts; gene_names : column names
    total_counts : (cells,) library size over ALL genes (denominator of CP10k)
    gene_lists : list of lists of gene names, one per protein
    Returns a dense (cells x len(gene_lists)) float32 array.
    """
    col = {g: i for i, g in enumerate(gene_names)}
    # Indicator matrix (genes x proteins): summing counts = counts @ indicator
    rows, cols = [], []
    for j, genes in enumerate(gene_lists):
        for g in genes:
            rows.append(col[g]); cols.append(j)
    ind = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(gene_names), len(gene_lists)))
    summed = np.asarray((counts @ ind).todense(), dtype=np.float64)
    cp10k = summed / np.asarray(total_counts, dtype=np.float64)[:, None] * target_sum
    return np.log1p(cp10k).astype(np.float32)


def pearson_columns(a, b):
    """Pearson r between matching columns of two (n x k) arrays.

    Returns NaN for a column where either input is constant.
    """
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    num = (a * b).sum(axis=0)
    den = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r[den == 0] = np.nan
    return r


def within_group_pearson(rna, prot, groups, min_cells):
    """Pearson r per protein inside each group with >= min_cells cells.

    Returns a long DataFrame: group, protein index, n_cells, r.
    """
    groups = np.asarray(groups).astype(str)
    out = []
    for g in pd.unique(groups):
        sel = groups == g
        n = int(sel.sum())
        if n < min_cells:
            continue
        r = pearson_columns(rna[sel].astype(np.float64), prot[sel].astype(np.float64))
        out.append(pd.DataFrame({"group": g, "j": np.arange(rna.shape[1]), "n_cells": n, "r": r}))
    return pd.concat(out, ignore_index=True)


def summarize_within(long, n_proteins):
    """Per protein: unweighted mean, cell-weighted mean, median of within-group r.

    Groups where r is NaN (gene or protein constant in that group) are skipped
    and counted in `n_groups_nan`.
    """
    rows = []
    for j in range(n_proteins):
        d = long[long["j"] == j]
        ok = d["r"].notna()
        dd = d[ok]
        rows.append({
            "mean": dd["r"].mean() if len(dd) else np.nan,
            "weighted_mean": np.average(dd["r"], weights=dd["n_cells"]) if len(dd) else np.nan,
            "median": dd["r"].median() if len(dd) else np.nan,
            "n_groups": int(ok.sum()),
            "n_groups_nan": int((~ok).sum()),
        })
    return pd.DataFrame(rows)


def between_group_pearson(rna, prot, groups, min_cells):
    """Pearson r between group means (pseudo-bulk), groups with >= min_cells cells."""
    df_r = pd.DataFrame(rna).groupby(np.asarray(groups).astype(str))
    sizes = df_r.size()
    keep = sizes.index[sizes >= min_cells]
    mr = df_r.mean().loc[keep].to_numpy()
    mp = pd.DataFrame(prot).groupby(np.asarray(groups).astype(str)).mean().loc[keep].to_numpy()
    return pearson_columns(mr, mp), len(keep)

"""Protein vs. same-gene RNA correlation, overall and within groups (Phase 2)."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata
from sklearn.utils.sparsefuncs import mean_variance_axis


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

    Returns NaN for a column where either input is constant. Constancy is checked
    exactly (max == min) before centering, because centering a constant column
    can leave tiny floating-point residues that would give a meaningless r.
    """
    const = (a.max(axis=0) == a.min(axis=0)) | (b.max(axis=0) == b.min(axis=0))
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    num = (a * b).sum(axis=0)
    den = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r[(den == 0) | const] = np.nan
    return r


def corr_columns(a, b, method="pearson"):
    """Column-wise correlation; Spearman = Pearson on ranks (ties get average ranks)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if method == "spearman":
        a, b = rankdata(a, axis=0), rankdata(b, axis=0)
    elif method != "pearson":
        raise ValueError(method)
    return pearson_columns(a, b)


def within_group_pearson(rna, prot, groups, min_cells, method="pearson"):
    """Correlation per protein inside each group with >= min_cells cells.

    For Spearman, ranks are computed inside each group.
    Returns a long DataFrame: group, protein index, n_cells, r.
    """
    groups = np.asarray(groups).astype(str)
    out = []
    for g in pd.unique(groups):
        sel = groups == g
        n = int(sel.sum())
        if n < min_cells:
            continue
        r = corr_columns(rna[sel], prot[sel], method)
        out.append(pd.DataFrame({"group": g, "j": np.arange(rna.shape[1]), "n_cells": n, "r": r}))
    if not out:  # no group reached min_cells
        return pd.DataFrame({"group": pd.Series(dtype=str), "j": pd.Series(dtype=int),
                             "n_cells": pd.Series(dtype=int), "r": pd.Series(dtype=float)})
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


def correlate_with_all_genes(X, Y):
    """Pearson r between every gene (columns of sparse X) and every column of Y.

    X : (cells x genes) sparse, e.g. log-normalized RNA
    Y : (cells x k) dense, e.g. protein CLR values
    Returns (genes x k) array. Uses cov = X^T (Y - mean(Y)) / n, so X is never
    densified. Genes with zero variance get NaN.
    """
    n = X.shape[0]
    Y = np.asarray(Y, dtype=np.float64)
    Yc = Y - Y.mean(axis=0)
    cov = np.asarray(X.T @ Yc) / n
    var_x = mean_variance_axis(X, axis=0)[1]
    sd_x = np.sqrt(np.asarray(var_x, dtype=np.float64))
    sd_y = Yc.std(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / sd_x[:, None] / sd_y[None, :]
    r[sd_x == 0, :] = np.nan
    return r


def mapped_gene_rank(r_all_genes, gene_names, mapped_r, names):
    """Where does the mapped gene's r fall among all genes, per protein?

    rank 1 = the mapped RNA correlates better with the protein than any other gene.
    Diagnostic only; must not be used to choose mappings.
    """
    rows = []
    for j, name in enumerate(names):
        col = r_all_genes[:, j]
        top = int(np.nanargmax(col))
        rows.append({
            "protein": name,
            "mapped_r": mapped_r[j],
            "mapped_rank": int(np.sum(col > mapped_r[j]) + 1) if np.isfinite(mapped_r[j]) else np.nan,
            "top_gene": gene_names[top],
            "top_gene_r": col[top],
        })
    return pd.DataFrame(rows)


def within_group_gene_correlation(X, Y, groups, min_cells):
    """Cell-weighted mean over groups of the gene x protein Pearson r.

    For each group with >= min_cells cells, r is computed inside the group with
    `correlate_with_all_genes`. Pairs that are undefined in a group (constant gene
    or protein there) are left out of that pair's average.
    Returns (mean_r, n_groups_used): mean_r is (genes x k).
    """
    groups = np.asarray(groups).astype(str)
    num = np.zeros((X.shape[1], Y.shape[1]))
    den = np.zeros_like(num)
    n_used = 0
    for g in pd.unique(groups):
        sel = np.flatnonzero(groups == g)
        if sel.size < min_cells:
            continue
        r = correlate_with_all_genes(X[sel], np.asarray(Y)[sel])
        ok = np.isfinite(r)
        num[ok] += sel.size * r[ok]
        den[ok] += sel.size
        n_used += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_r = num / den
    mean_r[den == 0] = np.nan
    return mean_r, n_used

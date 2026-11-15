import numpy as np
import pandas as pd
import scipy.sparse as sp

from citeseq.correlation import (between_group_pearson, mapped_log_expression,
                                 pearson_columns, summarize_within, within_group_pearson)


def test_pearson_columns_matches_numpy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(200, 3)), rng.normal(size=(200, 3))
    b[:, 0] += 2 * a[:, 0]
    r = pearson_columns(a, b)
    expected = [np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(3)]
    np.testing.assert_allclose(r, expected, rtol=1e-12)


def test_pearson_constant_column_is_nan():
    a = np.c_[np.ones(10), np.arange(10.0)]
    b = np.c_[np.arange(10.0), np.arange(10.0)]
    r = pearson_columns(a, b)
    assert np.isnan(r[0]) and np.isclose(r[1], 1.0)


def test_mapped_log_expression_sums_counts_before_log():
    counts = sp.csr_matrix(np.array([[1, 2, 7], [0, 0, 10]], dtype=float))
    total = np.array([10.0, 10.0])
    out = mapped_log_expression(counts, ["A", "B", "C"], total, [["A", "B"], ["C"]], target_sum=10)
    np.testing.assert_allclose(out, np.log1p([[3, 7], [0, 10]]), rtol=1e-6)


def test_within_and_between_group():
    # Two groups with very different means but no within-group relation:
    # strong overall correlation, ~0 within groups.
    rng = np.random.default_rng(1)
    n = 500
    groups = np.repeat(["g1", "g2"], n)
    rna = np.r_[rng.normal(0, 1, n), rng.normal(5, 1, n)][:, None]
    prot = np.r_[rng.normal(0, 1, n), rng.normal(5, 1, n)][:, None]
    overall = pearson_columns(rna, prot)[0]
    long = within_group_pearson(rna, prot, groups, min_cells=10)
    within = summarize_within(long, 1).loc[0]
    assert overall > 0.8
    assert abs(within["mean"]) < 0.15 and within["n_groups"] == 2
    r_between, k = between_group_pearson(rna, prot, groups, min_cells=10)
    assert k == 2 and np.isclose(r_between[0], 1.0)


def test_small_groups_are_skipped():
    rna = np.arange(30.0)[:, None]
    groups = np.array(["big"] * 25 + ["small"] * 5)
    long = within_group_pearson(rna, rna, groups, min_cells=10)
    assert set(long["group"]) == {"big"}


def test_spearman_is_rank_based():
    from citeseq.correlation import corr_columns
    x = np.arange(1.0, 21.0)[:, None]
    y = np.exp(x)  # monotone but non-linear
    assert np.isclose(corr_columns(x, y, "spearman")[0], 1.0)
    assert corr_columns(x, y, "pearson")[0] < 0.9


def test_correlate_with_all_genes_matches_dense():
    from citeseq.correlation import correlate_with_all_genes, mapped_gene_rank
    rng = np.random.default_rng(2)
    X = sp.random(300, 20, density=0.3, format="csr", random_state=3)
    X[:, 5] = 0  # constant gene
    Y = rng.normal(size=(300, 2))
    Y[:, 0] += 3 * X[:, 7].toarray().ravel()
    r = correlate_with_all_genes(X, Y)
    D = X.toarray()
    for g in range(20):
        for k in range(2):
            if g == 5:
                assert np.isnan(r[g, k])
            else:
                assert np.isclose(r[g, k], np.corrcoef(D[:, g], Y[:, k])[0, 1])
    diag = mapped_gene_rank(r, [f"g{i}" for i in range(20)], r[[7, 0], [0, 1]], ["p0", "p1"])
    assert diag.loc[0, "mapped_rank"] == 1 and diag.loc[0, "top_gene"] == "g7"


def test_within_group_gene_correlation_weighted_mean():
    from citeseq.correlation import within_group_gene_correlation
    rng = np.random.default_rng(4)
    X = sp.random(400, 6, density=0.5, format="csr", random_state=5)
    Y = rng.normal(size=(400, 2))
    groups = np.array(["a"] * 100 + ["b"] * 280 + ["c"] * 20)
    mean_r, n_used = within_group_gene_correlation(X, Y, groups, min_cells=50)
    assert n_used == 2  # group "c" is too small
    D = X.toarray()
    for g in range(6):
        ra = np.corrcoef(D[:100, g], Y[:100, 0])[0, 1]
        rb = np.corrcoef(D[100:380, g], Y[100:380, 0])[0, 1]
        assert np.isclose(mean_r[g, 0], (100 * ra + 280 * rb) / 380)

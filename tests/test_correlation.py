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

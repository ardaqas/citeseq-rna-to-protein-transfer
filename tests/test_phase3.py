import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import Ridge

from citeseq.baselines import CellTypeMean, GlobalMean, HVGPCAFeatures, OwnGeneLinear, RidgePath
from citeseq.metrics import evaluate, headline
from citeseq.splits import load_split, make_random_split, save_split


def _obs(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"cell_type": rng.choice(["a", "b", "c"], n, p=[0.6, 0.3, 0.1]),
                         "batch": rng.choice(["s1", "s2"], n)},
                        index=[f"cell{i}" for i in range(n)])


# ---------- splits ----------
def test_split_disjoint_complete_deterministic(tmp_path):
    obs = _obs()
    s1, s2 = make_random_split(obs, seed=0), make_random_split(obs, seed=0)
    assert (s1 == s2).all()
    assert set(s1.unique()) == {"train", "val", "test"} and len(s1) == len(obs)
    frac = s1.value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.01 and abs(frac["val"] - 0.15) < 0.01
    # stratification: every cell type appears in every part
    assert pd.crosstab(obs["cell_type"], s1).min().min() > 0
    save_split(s1, tmp_path / "split.csv.gz")
    back = load_split(tmp_path / "split.csv.gz", obs.index[::-1])
    assert (back == s1.loc[obs.index[::-1]]).all()


# ---------- metrics ----------
def test_constant_prediction_gives_nan_not_zero():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(300, 2))
    ct = np.repeat(["a", "b", "c"], 100)
    m = evaluate(y, GlobalMean().fit(y).predict(300), ct, ["p1", "p2"], min_cells=50)
    assert m["pearson_all"].isna().all() and m["within_w"].isna().all()
    assert (m["n_types_nan"] == 3).all()
    np.testing.assert_allclose(m["rmse"], y.std(axis=0), rtol=1e-12)
    h = headline(m, ["p1", "p2"])
    assert np.isnan(h.loc["pearson_all", "median"]) and h.loc["pearson_all", "n_defined"] == 0


def test_celltype_mean_within_undefined_overall_defined():
    rng = np.random.default_rng(2)
    ct = np.repeat(["a", "b"], 200)
    y = (ct == "b")[:, None] * 3.0 + rng.normal(size=(400, 1))
    pred = CellTypeMean().fit(y, ct).predict(ct)
    m = evaluate(y, pred, ct, ["p"], min_cells=50)
    assert m.loc["p", "pearson_all"] > 0.7 and np.isnan(m.loc["p", "within_w"])


def test_weighted_within_mean():
    rng = np.random.default_rng(3)
    ct = np.array(["a"] * 100 + ["b"] * 300)
    pred = rng.normal(size=(400, 1))
    y = pred + rng.normal(size=(400, 1))
    m = evaluate(y, pred, ct, ["p"], min_cells=50)
    ra = np.corrcoef(pred[:100, 0], y[:100, 0])[0, 1]
    rb = np.corrcoef(pred[100:, 0], y[100:, 0])[0, 1]
    assert np.isclose(m.loc["p", "within_w"], (100 * ra + 300 * rb) / 400)
    assert np.isclose(m.loc["p", "within_unw"], (ra + rb) / 2)


# ---------- models ----------
def test_celltype_mean_equals_training_means():
    ct = np.array(["a", "a", "b", "b", "b"])
    y = np.array([[1.0], [3.0], [10.0], [11.0], [12.0]])
    model = CellTypeMean().fit(y, ct)
    np.testing.assert_allclose(model.predict(["a", "b", "zzz"]).ravel(), [2.0, 11.0, y.mean()])
    assert model.n_fallback_ == 1


def test_own_gene_equals_ols_and_nan_column():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(200, 2))
    y = 2.0 + 0.5 * x + rng.normal(scale=0.1, size=(200, 2))
    x[:, 1] = np.nan
    model = OwnGeneLinear().fit(x, y)
    b, a = np.polyfit(x[:, 0], y[:, 0], 1)
    assert np.isclose(model.slope_[0], b) and np.isclose(model.intercept_[0], a)
    assert np.isnan(model.predict(x)[:, 1]).all()


def test_ridge_path_matches_sklearn():
    rng = np.random.default_rng(5)
    X, Y = rng.normal(size=(150, 8)), rng.normal(size=(150, 3))
    path = RidgePath().fit(X, Y, alphas=[0.1, 10.0])
    for a in (0.1, 10.0):
        ref = Ridge(alpha=a).fit(X, Y).predict(X)
        np.testing.assert_allclose(path.predict(X, a), ref, rtol=1e-8, atol=1e-10)


def test_feature_pipeline_fit_on_train_only():
    rng = np.random.default_rng(6)
    n, g = 400, 300
    X = sp.csr_matrix(np.log1p(rng.poisson(rng.gamma(0.5, 2.0, size=g), size=(n, g)).astype(float)))
    obs = _obs(n)
    tr = np.arange(n) < 300
    f = HVGPCAFeatures(n_top_genes=50, max_pcs=10).fit(X[tr], obs[tr], [f"g{i}" for i in range(g)])
    state = (f.hvg_mask_.copy(), f.pca_.components_.copy(), f.scaler_.mean_.copy())
    f.transform_pca(X[~tr], 5)
    f.transform_hvg(X[~tr])
    assert (f.hvg_mask_ == state[0]).all()
    np.testing.assert_array_equal(f.pca_.components_, state[1])
    np.testing.assert_array_equal(f.scaler_.mean_, state[2])
    assert f.hvg_mask_.sum() == 50 and f.transform_pca(X[~tr], 5).shape == (100, 5)


def test_split_handles_very_rare_labels():
    obs = _obs()
    obs.iloc[:2, obs.columns.get_loc("cell_type")] = ["only_one", "another_single"]
    s = make_random_split(obs, seed=0)
    assert len(s) == len(obs) and set(s.unique()) == {"train", "val", "test"}


def test_ridge_full_rank_pca_equals_ridge_on_unscaled_hvg():
    # Ridge is invariant to orthogonal rotations of centered inputs, and full-rank PCA
    # is such a rotation, so both must give the same predictions for the same alpha.
    rng = np.random.default_rng(7)
    n, g = 500, 200
    X = sp.csr_matrix(np.log1p(rng.poisson(rng.gamma(0.8, 2.0, size=g), size=(n, g)).astype(float)))
    obs = _obs(n)
    Y = rng.normal(size=(n, 3))
    tr = np.arange(n) < 400
    f = HVGPCAFeatures(n_top_genes=40, max_pcs=None).fit(X[tr], obs[tr], [f"g{i}" for i in range(g)])
    k = int(f.hvg_mask_.sum())
    assert f.pca_.n_components_ == k
    for alpha in (0.5, 50.0):
        p_pca = RidgePath().fit(f.transform_pca(X[tr], k), Y[tr], [alpha]).predict(f.transform_pca(X[~tr], k), alpha)
        p_hvg = RidgePath().fit(f.hvg_dense(X[tr]), Y[tr], [alpha]).predict(f.hvg_dense(X[~tr]), alpha)
        np.testing.assert_allclose(p_pca, p_hvg, atol=1e-4)

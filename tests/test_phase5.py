import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from citeseq.phase5 import fit_predict_fold, group_bias, shift_captured, weighted_abs_bias
from citeseq.splits import group_folds, inner_split

torch.set_num_threads(2)
CFG3 = {"features": {"hvg": 30}, "seed": 0, "3a_ridge_pca": {"n_pcs": 30, "alpha": 10.0}}
CFG4 = {"best": {"hidden": [16], "dropout": 0.1, "weight_decay": 1e-4, "lr": 1e-3, "batch_size": 64}}


def _data(n=600, g=120, k=4, seed=0):
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(np.log1p(rng.poisson(rng.gamma(0.8, 2.0, size=g), size=(n, g)).astype(float)))
    obs = pd.DataFrame({"cell_type": rng.choice(["a", "b", "c"], n), "batch": rng.choice(["s1", "s2", "s3"], n)},
                       index=[f"c{i}" for i in range(n)])
    obs["DonorNumber"] = obs["batch"]
    Y = rng.normal(size=(n, k)) + X[:, :k].toarray()
    Rm = X[:, :k].toarray().astype(np.float32)
    return X, Y, Rm, obs, np.array([f"g{i}" for i in range(g)])


def test_test_targets_never_used():
    X, Y, Rm, obs, genes = _data()
    test = (obs["batch"] == "s3").to_numpy()
    train = ~test
    Y2 = Y.copy()
    Y2[test] = np.random.default_rng(9).normal(size=(test.sum(), Y.shape[1])) * 100
    p1, s1 = fit_predict_fold(X, Y, Rm, obs, genes, train, test, CFG3, CFG4, max_epochs=3, patience=2)
    p2, s2 = fit_predict_fold(X, Y2, Rm, obs, genes, train, test, CFG3, CFG4, max_epochs=3, patience=2)
    for m in p1:
        np.testing.assert_allclose(p1[m], p2[m], rtol=1e-6, err_msg=m)
    # target scaler = mean of the MLP's training cells only (inner 85% of train)
    tr_idx = np.flatnonzero(train)
    m_in, _ = inner_split(obs["cell_type"].to_numpy()[train], frac=0.15, seed=0)
    np.testing.assert_allclose(s1["y_scaler_mean"], Y[tr_idx[m_in]].mean(axis=0))
    assert s1["n_mlp_fit"] + s1["n_mlp_stop"] == train.sum()
    np.testing.assert_allclose(s1["sd_train"], Y[train].std(axis=0))


def test_feature_pipeline_ignores_test_inputs():
    X, Y, Rm, obs, genes = _data(seed=1)
    test = (obs["batch"] == "s1").to_numpy()
    X2 = X.tolil(); X2[np.flatnonzero(test)[:50], :] = 5.0; X2 = X2.tocsr()
    _, s1 = fit_predict_fold(X, Y, Rm, obs, genes, ~test, test, CFG3, CFG4, max_epochs=2, patience=1)
    _, s2 = fit_predict_fold(X2, Y, Rm, obs, genes, ~test, test, CFG3, CFG4, max_epochs=2, patience=1)
    assert (s1["hvg_mask"] == s2["hvg_mask"]).all()
    np.testing.assert_allclose(s1["x_scaler_mean"], s2["x_scaler_mean"])


def test_group_folds_and_inner_split():
    obs = pd.DataFrame({"Site": ["s2", "s1", "s1", "s3"]})
    f = group_folds(obs, "Site")
    assert list(f) == ["s1", "s2", "s3"] and f["s1"].tolist() == [False, True, True, False]
    m_tr, m_va = inner_split(np.repeat(["a", "b"], 100), frac=0.15, seed=0)
    assert m_va.sum() == 30 and not (m_tr & m_va).any()


def test_bias_and_shift_captured_ratio_of_sums():
    y = np.zeros((200, 1)); groups = np.repeat(["g1", "g2"], [50, 150]); sd = np.array([2.0])
    ref = np.r_[np.full(50, 2.0), np.full(150, -4.0)][:, None]      # |b| = 1 and 2
    mod = np.r_[np.full(50, 1.0), np.full(150, -1.0)][:, None]      # |b| = 0.5 and 0.5
    lr, lm = group_bias(y, ref, groups, sd, 10), group_bias(y, mod, groups, sd, 10)
    assert np.isclose(weighted_abs_bias(lr), (50 * 1 + 150 * 2) / 200)
    # ratio of sums: 1 - (50*.5 + 150*.5) / (50*1 + 150*2) = 1 - 100/350
    assert np.isclose(shift_captured(lm, lr), 1 - 100 / 350)
    assert len(group_bias(y, mod, groups, sd, 100)) == 1  # g1 (50 cells) dropped

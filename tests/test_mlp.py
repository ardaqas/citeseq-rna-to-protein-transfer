import numpy as np
import torch

from citeseq.correlation import pearson_columns
from citeseq.mlp import TargetScaler, predict, train_mlp

torch.set_num_threads(2)


def _data(seed=0, n=1200, d=20, k=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    W = rng.normal(size=(d, k))
    Y = np.tanh(X @ W) * 2.0 + 5.0 + 0.1 * rng.normal(size=(n, k))
    return X[:900], Y[:900], X[900:], Y[900:]


def test_target_scaler_roundtrip():
    Y = np.random.default_rng(1).normal(3, 2, size=(50, 4))
    s = TargetScaler().fit(Y)
    np.testing.assert_allclose(s.inverse(s.transform(Y)), Y, rtol=1e-5)


def test_mlp_learns():
    Xtr, Ytr, Xva, Yva = _data()
    s = TargetScaler().fit(Ytr)
    out = train_mlp(Xtr, s.transform(Ytr), Xva, s.transform(Yva), hidden=[32], dropout=0.0,
                    weight_decay=0.0, seed=0, max_epochs=300, patience=10)
    r = pearson_columns(predict(out["model"], Xva, s), Yva)
    assert np.all(r > 0.8)  # non-linear target, 900 training samples: ~0.86 expected
    assert out["best_epoch"] == int(np.argmin(out["history"]["val_loss"]))


def test_mlp_early_stops_on_noise():
    # Targets unrelated to inputs: validation loss cannot keep improving, so training must stop early.
    rng = np.random.default_rng(5)
    Xtr, Xva = rng.normal(size=(600, 20)).astype(np.float32), rng.normal(size=(300, 20)).astype(np.float32)
    Ztr, Zva = rng.normal(size=(600, 2)).astype(np.float32), rng.normal(size=(300, 2)).astype(np.float32)
    out = train_mlp(Xtr, Ztr, Xva, Zva, hidden=[64], dropout=0.0, weight_decay=0.0, seed=0,
                    batch_size=32, max_epochs=300, patience=5)
    h = out["history"]
    assert len(h["val_loss"]) < 300
    assert len(h["val_loss"]) == out["best_epoch"] + 1 + 5  # stopped exactly `patience` epochs after the best


def test_same_seed_same_predictions_and_checkpoint_reuse(tmp_path):
    Xtr, Ytr, Xva, Yva = _data(2)
    s = TargetScaler().fit(Ytr)
    args = dict(hidden=[16], dropout=0.1, weight_decay=1e-4, max_epochs=20, patience=5)
    a = train_mlp(Xtr, s.transform(Ytr), Xva, s.transform(Yva), seed=3, checkpoint=tmp_path / "a.pt", **args)
    b = train_mlp(Xtr, s.transform(Ytr), Xva, s.transform(Yva), seed=3, **args)
    np.testing.assert_allclose(predict(a["model"], Xva, s), predict(b["model"], Xva, s), rtol=1e-6)
    c = train_mlp(Xtr, s.transform(Ytr), Xva, s.transform(Yva), seed=3, checkpoint=tmp_path / "a.pt", **args)
    assert c["loaded"] and not a["loaded"]
    np.testing.assert_allclose(predict(a["model"], Xva, s), predict(c["model"], Xva, s), rtol=1e-6)
    d = train_mlp(Xtr, s.transform(Ytr), Xva, s.transform(Yva), seed=4, checkpoint=tmp_path / "a.pt", **args)
    assert not d["loaded"]  # different config -> retrained, not loaded

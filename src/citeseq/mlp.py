"""Small multi-output MLP (PyTorch, CPU) for Phase 4.

* Inputs: z-scored HVG matrix (scaler fit on training cells only).
* Targets: protein CLR values standardized per protein with training mean/SD, so
  every protein carries equal weight in the MSE loss. Predictions are mapped back.
* Early stopping on validation loss; the best-epoch weights are kept.
* Each finished run is saved as a checkpoint; calling `train_mlp` again with the
  same checkpoint path loads it instead of retraining (resumable after a crash).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden, dropout):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TargetScaler:
    """Per-protein standardization fit on training targets."""

    def fit(self, Y):
        Y = np.asarray(Y, dtype=np.float64)
        self.mean_ = Y.mean(axis=0)
        self.sd_ = Y.std(axis=0)
        self.sd_[self.sd_ == 0] = 1.0
        return self

    def transform(self, Y):
        return ((np.asarray(Y, dtype=np.float64) - self.mean_) / self.sd_).astype(np.float32)

    def inverse(self, Z):
        return np.asarray(Z, dtype=np.float64) * self.sd_ + self.mean_


def _predict_std(model, X, batch_size=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out.append(model(X[i:i + batch_size]).numpy())
    return np.vstack(out)


def train_mlp(X_tr, Z_tr, X_va, Z_va, hidden, dropout, weight_decay, seed, lr=1e-3,
              batch_size=512, max_epochs=200, patience=15, checkpoint=None, verbose=False):
    """Train one MLP with early stopping on validation MSE (standardized targets).

    X_* : float32 inputs; Z_* : standardized float32 targets.
    Returns dict(model, history, best_epoch, config). `best_epoch` is a 0-based index
    into history (1-based epoch = best_epoch + 1); training stops `patience` epochs
    after it, so len(history) == best_epoch + 1 + patience unless max_epochs is hit.
    With `checkpoint`, a finished run is saved there and re-used on the next call.
    """
    config = dict(hidden=list(hidden), dropout=dropout, weight_decay=weight_decay, seed=seed,
                  lr=lr, batch_size=batch_size, max_epochs=max_epochs, patience=patience)
    model = MLP(X_tr.shape[1], Z_tr.shape[1], hidden, dropout)
    if checkpoint is not None and Path(checkpoint).exists():
        ck = torch.load(checkpoint)
        if ck["config"] == config:
            model.load_state_dict(ck["state"])
            return dict(model=model, history=ck["history"], best_epoch=ck["best_epoch"], config=config, loaded=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    Xt, Zt = torch.from_numpy(np.ascontiguousarray(X_tr)), torch.from_numpy(np.ascontiguousarray(Z_tr))
    Xv, Zv = torch.from_numpy(np.ascontiguousarray(X_va)), torch.from_numpy(np.ascontiguousarray(Z_va))
    model = MLP(X_tr.shape[1], Z_tr.shape[1], hidden, dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history = {"train_loss": [], "val_loss": []}
    best, best_epoch, best_state, wait = np.inf, -1, None, 0
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        total = 0.0
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), Zt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        val_loss = float(np.mean((_predict_std(model, Xv) - Zv.numpy()) ** 2))
        history["train_loss"].append(total / len(Xt))
        history["val_loss"].append(val_loss)
        if verbose:
            print(f"epoch {epoch:3d}  train {history['train_loss'][-1]:.4f}  val {val_loss:.4f}")
        if val_loss < best - 1e-6:
            best, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    if checkpoint is not None:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": config, "state": best_state, "history": history, "best_epoch": best_epoch}, checkpoint)
    return dict(model=model, history=history, best_epoch=best_epoch, config=config, loaded=False)


def predict(model, X, target_scaler):
    """Predictions on the original (CLR) scale."""
    return target_scaler.inverse(_predict_std(model, torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))))

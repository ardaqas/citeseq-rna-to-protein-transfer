"""Baseline models for Phase 3 and the train-only feature pipeline.

Everything that learns from data (means, slopes, HVG selection, scaling, PCA,
ridge weights) is fit on training cells only. Validation/test cells only go
through `predict` / `transform`.
"""
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class GlobalMean:
    """Predict each protein's training mean for every cell."""

    def fit(self, Y):
        self.mean_ = np.asarray(Y, dtype=np.float64).mean(axis=0)
        return self

    def predict(self, n_cells):
        return np.tile(self.mean_, (n_cells, 1))


class CellTypeMean:
    """Predict each protein's training mean within the cell's annotated type.

    Uses the cell-type LABEL as input, which a model from RNA alone would not have.
    Types unseen in training fall back to the global mean (counted in `n_fallback_`).
    """

    def fit(self, Y, cell_type):
        Y = np.asarray(Y, dtype=np.float64)
        self.global_mean_ = Y.mean(axis=0)
        self.means_ = pd.DataFrame(Y).groupby(np.asarray(cell_type).astype(str)).mean()
        return self

    def predict(self, cell_type):
        ct = pd.Index(np.asarray(cell_type).astype(str))
        known = ct.isin(self.means_.index)
        self.n_fallback_ = int((~known).sum())
        out = np.tile(self.global_mean_, (len(ct), 1))
        out[known] = self.means_.loc[ct[known]].to_numpy()
        return out


class OwnGeneLinear:
    """Per protein: y = a + b * x, where x is the RNA of the protein's own gene(s).

    Columns of x that are entirely NaN (protein without a measured gene) give NaN
    predictions.
    """

    def fit(self, x, Y):
        x = np.asarray(x, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        xm, ym = x.mean(axis=0), Y.mean(axis=0)
        xc, yc = x - xm, Y - ym
        with np.errstate(invalid="ignore", divide="ignore"):
            self.slope_ = (xc * yc).sum(axis=0) / (xc * xc).sum(axis=0)
        self.intercept_ = ym - self.slope_ * xm
        return self

    def predict(self, x):
        return self.intercept_ + self.slope_ * np.asarray(x, dtype=np.float64)


class RidgePath:
    """Multi-output ridge regression for several alphas from one X^T X.

    Equivalent to sklearn Ridge(alpha, fit_intercept=True) for each alpha, but the
    Gram matrix is computed once, which makes an alpha grid cheap.
    """

    def fit(self, X, Y, alphas):
        Xc = np.array(X, dtype=np.float64, copy=True)  # one float64 copy, centered in place
        Y = np.asarray(Y, dtype=np.float64)
        self.x_mean_, self.y_mean_ = Xc.mean(axis=0), Y.mean(axis=0)
        Xc -= self.x_mean_
        self.gram_ = Xc.T @ Xc
        self.xty_ = Xc.T @ (Y - self.y_mean_)
        self.coef_ = {}
        return self.extend(alphas)

    def extend(self, alphas):
        """Add solutions for more alphas without recomputing X^T X."""
        eye = np.eye(self.gram_.shape[0])
        for a in alphas:
            if a not in self.coef_:
                self.coef_[a] = np.linalg.solve(self.gram_ + a * eye, self.xty_)
        return self

    def predict(self, X, alpha):
        return (np.asarray(X, dtype=np.float64) - self.x_mean_) @ self.coef_[alpha] + self.y_mean_


class HVGPCAFeatures:
    """Train-only feature pipeline on log-normalized RNA.

    fit():  HVGs chosen per sample on training cells (scanpy, flavor="seurat");
            a StandardScaler and a PCA fit on the training HVG matrix.
            max_pcs=None keeps all components (full rank = number of HVGs).
    transform_pca(X, n_pcs): PCA scores (first n_pcs components).
    transform_hvg(X):        z-scored HVG matrix (dense float32).
    hvg_dense(X):            unscaled log-normalized HVG matrix (dense float32).
    """

    def __init__(self, n_top_genes=2000, max_pcs=200, seed=0):
        self.n_top_genes, self.max_pcs, self.seed = n_top_genes, max_pcs, seed

    def fit(self, X_train, obs_train, var_names, batch_key="batch"):
        tmp = ad.AnnData(X=X_train, obs=obs_train[[batch_key]].copy(),
                         var=pd.DataFrame(index=pd.Index(var_names)))
        hv = sc.pp.highly_variable_genes(tmp, n_top_genes=self.n_top_genes, batch_key=batch_key,
                                         flavor="seurat", inplace=False)
        self.hvg_mask_ = hv["highly_variable"].to_numpy()
        self.hvg_names_ = np.asarray(var_names)[self.hvg_mask_]
        Xh = X_train[:, self.hvg_mask_]
        # PCA on the sparse log-normalized HVG matrix (centering is handled by sklearn)
        n_comp = int(self.hvg_mask_.sum()) if self.max_pcs is None else self.max_pcs
        self.pca_ = PCA(n_components=n_comp, svd_solver="covariance_eigh",
                        random_state=self.seed).fit(Xh)
        self.scaler_ = StandardScaler().fit(Xh.toarray())
        return self

    def transform_pca(self, X, n_pcs):
        return self.pca_.transform(X[:, self.hvg_mask_])[:, :n_pcs].astype(np.float32)

    def hvg_dense(self, X):
        return X[:, self.hvg_mask_].toarray().astype(np.float32)

    def transform_hvg(self, X):
        return self.scaler_.transform(X[:, self.hvg_mask_].toarray()).astype(np.float32)

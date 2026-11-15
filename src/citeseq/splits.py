"""Train/validation/test splits, saved to disk so every phase uses the same one."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def make_random_split(obs, seed=0, fracs=(0.70, 0.15, 0.15), stratify="cell_type", min_per_class=10):
    """Random cell-level split, stratified by `stratify` so every label occurs in every part.

    Labels with fewer than `min_per_class` cells are pooled into one stratum for the
    stratification only (stratifying needs a few cells per label). On the full data
    the smallest cell type has 18 cells, so no label is pooled there.
    Returns a Series (index = obs.index) with values "train" / "val" / "test".
    """
    if not np.isclose(sum(fracs), 1.0):
        raise ValueError("fracs must sum to 1")
    idx = np.arange(len(obs))
    y = obs[stratify].astype(str).to_numpy()
    counts = pd.Series(y).value_counts()
    rare = counts.index[counts < min_per_class]
    y = np.where(np.isin(y, rare), "__rare__", y)
    if 0 < (y == "__rare__").sum() < min_per_class:  # pooled stratum still too small
        y = np.where(y == "__rare__", counts.index[0], y)
    train, rest = train_test_split(idx, train_size=fracs[0], stratify=y, random_state=seed)
    val, test = train_test_split(rest, train_size=fracs[1] / (fracs[1] + fracs[2]),
                                 stratify=y[rest], random_state=seed)
    split = pd.Series("train", index=obs.index, name="split")
    split.iloc[val] = "val"
    split.iloc[test] = "test"
    return split


def save_split(split, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    split.rename_axis("cell").to_csv(path)


def load_split(path, obs_names):
    """Load a saved split and align it to `obs_names` (every cell must be present)."""
    s = pd.read_csv(path, index_col=0)["split"]
    missing = pd.Index(obs_names).difference(s.index)
    if len(missing):
        raise KeyError(f"{len(missing)} cells are not in the saved split")
    return s.loc[obs_names]

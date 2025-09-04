import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer, accuracy_score, f1_score, recall_score, precision_score

# Clean data - substitui '?' por NaN
class CleanNumeric(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            Z = X.replace('?', np.nan)
            Z = Z.apply(pd.to_numeric, errors='coerce')
            return Z.values
        return X

# Seletor supervisionado por correlação (abs) com y
class CorrFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k: Optional[int] = None):
        self.k = k
        self.indices_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        corrs = []
        for j in range(X.shape[1]):
            xj = X[:, j]
            if np.all(np.isnan(xj)) or np.nanstd(xj) == 0 or np.nanstd(y) == 0:
                corrs.append(0.0)
            else:
                c = np.corrcoef(np.nan_to_num(xj, nan=np.nanmean(xj)), y)[0,1]
                if np.isnan(c):
                    c = 0.0
                corrs.append(abs(c))
        order = np.argsort(corrs)[::-1]
        self.indices_ = order[: self.k]
        return self

    def transform(self, X):
        if self.indices_ is None:
            return X
        X = np.asarray(X)
        return X[:, self.indices_]

# Configurações e scoring
@dataclass(frozen=True)
class CVConfig:
    n_splits: int = 10
    shuffle: bool = True
    random_state: int = 42

scorers = {
    'accuracy': make_scorer(accuracy_score),
    'f1':       make_scorer(f1_score),
    'recall':   make_scorer(recall_score),
    'precision':make_scorer(precision_score),
}
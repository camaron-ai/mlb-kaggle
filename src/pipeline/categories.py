from pandas.api.types import is_categorical_dtype
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer


class Categorify(BaseEstimator, TransformerMixin):
    def __init__(self, top_n: int = None,
                 add_nan: bool = False,
                 dtype=np.int64):
        self.top_n = top_n
        self.add_nan = add_nan
        self.dtype = dtype

    def fit(self, X: pd.Series):
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        has_nan = X.isna().sum() > 0
        self.add_nan = (self.add_nan and has_nan)
        categories = (list(X.categories)
                      if is_categorical_dtype(X)
                      else sorted(list(f for f in X.drop_duplicates()
                                       if not pd.isna(f))))
        if self.top_n is not None:
            counter = Counter(X)
            top_categories = [cat for cat, _ in
                              counter.most_common(self.top_n)]
            categories = [cat for cat in categories
                          if cat in top_categories]
        self.categories = categories
        return self

    def transform(self, X: pd.Series) -> pd.DataFrame:
        Xcat = pd.Categorical(X, categories=self.categories, ordered=True)
        Xcodes = Xcat.codes.astype(self.dtype)
        if self.add_nan:
            Xcodes += 1
        assert Xcodes.min() == 0, f'the min is {Xcodes.min()}'
        return pd.DataFrame({X.name: Xcodes})


class PdKBinsDiscretizer(KBinsDiscretizer):
    def __init__(self, n_bins=5,
                 encode: str = 'ordinal',
                 strategy: str ='quantile'):
        super().__init__(n_bins=n_bins,
                 encode=encode,
                 strategy=strategy)
        
    def transform(self, X: pd.DataFrame):
        features = list(X.columns)
        outputX = super().transform(X).astype(np.int64)
        return pd.DataFrame(outputX, columns=features)
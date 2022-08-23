from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from typing import List
from pandas.api.types import is_float_dtype


class MedianFillNaN(BaseEstimator, TransformerMixin):
    def __init__(self, features: List[str] = None):
        self.features = features

    def fit(self, df: pd.DataFrame, y=None):
        if self.features is not None:
            df = df.loc[:, self.features]
        self.medians = df.median().to_dict()
        return self
        
    def transform(self, X: pd.DataFrame):
        return X.fillna(self.medians)



class FilterContinuousFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, ignore_features: List[str]):
        self.ignore_features = ignore_features

    def fit(self, df: pd.DataFrame, y=None):
        self._features = [name for name, values in df.items()
                          if (is_float_dtype(values) and
                              name not in self.ignore_features)]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:        
        return df.loc[:, self._features]



class PdStandardNorm(BaseEstimator, TransformerMixin):
    eps = 1e-15

    def fit(self, X: pd.DataFrame , y=None):
        self.means = X.mean()
        self.stds = X.std() + self.eps
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        output = (X - self.means) / self.stds
        return output.astype(np.float16)


class PdScaleNorm(BaseEstimator, TransformerMixin):
    eps = 1e-15

    def fit(self, X: pd.DataFrame, y=None):
        self.min = X.min()
        self.max = X.max()
        self.difference = self.max - self.min + self.eps
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        output = (X - self.min) / self.difference
        return output.astype(np.float16)


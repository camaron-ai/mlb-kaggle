from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
import pandas as pd
import numpy as np


class StatisticTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        self._features = list(X.columns.drop(self.index_cols))
        return self
    
    def transform(self, X: pd.DataFrame):
        X.set_index('date', inplace=True)
        output = X.groupby(self.ids)[self._features].apply(self.compute_features)
        output.reset_index(inplace=True)
        
        X.reset_index(inplace=True)
        
        assert np.all(X[self.index_cols] == output[self.index_cols]), \
               'the ids do not match!'
        if self.drop_index:
            output.drop(self.index_cols, axis=1, inplace=True)
        return output


class StatisticGen(StatisticTransformer):
    def __init__(self, stats: List[str] = ['mean'],
                 windows: List[int] = [10],
                 dt_col: str = 'date',
                 ids: List[str] = ['playerId'],
                 drop_index: bool = True):        
        self.stats = stats
        self.windows = windows
        self.dt_col = dt_col
        self.ids = ids
        self.index_cols = self.ids + [self.dt_col]
        self.drop_index = drop_index
        
    def _compute_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        stats_df = df.rolling(window, min_periods=1).agg(self.stats)
        stats_df.columns = ['__'.join(list(f) + [f'{window}d'])
                            for f in stats_df]
        return stats_df.astype(np.float32)
    
    def compute_features(self, df: pd.DataFrame):
        return pd.concat([self._compute_features(df, window)
                               for window in self.windows], axis=1)
    


class LagGen(StatisticTransformer):
    def __init__(self, lags: List[int] = [10],
                 dt_col: str = 'date',
                 ids: List[str] = ['playerId'],
                 drop_index: bool = True):
        self.lags = lags
        self.dt_col = dt_col
        self.ids = ids
        self.index_cols = self.ids + [self.dt_col]
        self.drop_index = drop_index
        
    def _compute_features(self, df: pd.DataFrame, lag: int) -> pd.DataFrame:
        lagdf = df.shift(lag)
        lagdf.columns = [f'{f}__{lag}lag' for f in lagdf.columns]
        return lagdf.astype(np.float32)
        
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([self._compute_features(df, lag)
                          for lag in self.lags], axis=1) 


class FeaturesTable(BaseEstimator, TransformerMixin):
    def __init__(self, table: pd.DataFrame,
                 lags: List[int],
                 on: List[str] = ['playerId', 'date'],
                 date_col: str = 'date',
                 clip_max: bool = True):
        self.on = on
        self.lags = lags
        self.table = table
        self.date_col = date_col
        self.clip_max = clip_max
        self.max_date = table[self.date_col].max().to_numpy()

    
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def _merge_table(self, X: pd.DataFrame,
                     lag: int) -> pd.DataFrame:
        Xon = X.loc[:, self.on].copy()
        Xon[self.date_col] = X[self.date_col] - pd.to_timedelta(lag, unit='d')
        if self.clip_max:
            Xon[self.date_col] = np.minimum(Xon[self.date_col], self.max_date)
        outputX = Xon.merge(self.table, on=self.on,
                            how='left').drop(self.on, axis=1)
        assert len(outputX) == len(X), \
               f'the len {len(X)} of the input do not match the output len {len(outputX)}'
        outputX.columns += f'__{lag}lag'
        return outputX

    def transform(self, X: pd.DataFrame):
        return pd.concat([self._merge_table(X, lag)
                          for lag in self.lags], axis=1)
    
    
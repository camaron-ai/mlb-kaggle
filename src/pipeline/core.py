from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from typing import Union, List, Tuple
from functools import reduce


def pandas_hstack(Xs):
    return pd.concat([X.reset_index(drop=True) for X in Xs], axis=1)


def reduce_fn(left: pd.DataFrame,
              right: pd.DataFrame):
    common_cols = [f for f in left.columns
                   if f in right.columns]
    if len(common_cols) > 0:
        return left.merge(right, on=common_cols, how='left')
    return pd.concat([left, right], axis=1)



class CoreTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=np.ndarray):
        return self
    
    def transform(self, X: pd.DataFrame):
        return X


class PdColumnTransformer(ColumnTransformer):
    def _hstack(self, Xs):
        return reduce(reduce_fn, Xs)


class PdFeatureUnion(FeatureUnion):
    def _hstack(self, Xs):
        return reduce(reduce_fn, Xs)


def unpack_json(json_str):
    return np.nan if pd.isna(json_str) else pd.read_json(json_str)


def unpack_column(series: pd.DataFrame) -> pd.DataFrame:
    def _unpack_row(index, row):
        out_df = unpack_json(row)
        out_df = out_df.assign(date=index)
        return out_df
    
    return pd.concat([_unpack_row(index, row)
                      for index, row in series.iteritems()
                      if pd.notna(row)], ignore_index=True)


def make_column_tmf(*transformers: List[Tuple[TransformerMixin, Union[List[str], str]]],
                    **kwargs):
    names = [type(tmf).__name__ for (tmf, _) in transformers]
    return PdColumnTransformer([(name, tmf, features)
                                 for name, (tmf, features) in zip(names, transformers)], **kwargs)


def make_unpack_tmf(column_name: str):
    return make_column_tmf((FunctionTransformer(unpack_column), column_name))


def forward_fill(df: pd.DataFrame, features: List[str],
                 on='playerId',
                 suffix: str = None,
                 limit: int =None):
    ffilled_df = df.groupby(on)[features].ffill(limit=limit)
    output_features = (features if suffix is None else
                       list(map(lambda f: suffix + f, features)))
    outputX = df.copy()
    outputX.loc[:, output_features] = ffilled_df.to_numpy()
    return outputX


def gen_hardcoded_features(df: pd.DataFrame):
    # some feature eng for the dates
    df['pstatsTime'] = (df['date'] - pd.to_datetime(df['pstatsDate'])).dt.total_seconds()
    df['playerAge'] = (df['date'] - df['DOB']).dt.total_seconds()
    df['playerTSinceDebut'] = (df['date'] - df['mlbDebutDate']).dt.total_seconds()
    df['playerDebutAge'] = (df['mlbDebutDate'] - df['DOB']).dt.total_seconds()
    df['rostersTime'] = (df['date'] - pd.to_datetime(df['rosterDate'])).dt.total_seconds()
    df[['rostersTime', 'pstatsTime']] /= 60 * 60 * 24
    # normalize
    df[['playerAge', 'playerTSinceDebut', 'playerDebutAge']] /= 60 * 60 * 24 * 365

    # unify categories
    # teams = df['teamId'].unique()
    # team_from_mlb = (~df['MoveToTeamId'].isin(teams)) & df['MoveToTeamId'].notna()
    # df.loc[team_from_mlb, 'MoveToTeamId'] = 0
    return df

def fillna(df: pd.DataFrame, fill_value=0):
    return df.fillna(fill_value)


class FilterFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, suffixes: List[str]):
        self.suffixes = suffixes
    
    def fit(self, X: pd.DataFrame, y=None):
        self._features = list(filter(lambda f: any([f.endswith(suffix) for suffix in self.suffixes]),
                                     X.columns))
        return self
    
    def transform(self, X: pd.DataFrame):
        return X.loc[:, self._features]
import pandas as pd
from typing import List
import numpy as np


def filter_by_id(df: pd.DataFrame, ids: List[int]) -> pd.DataFrame:
    index = df['playerId'].isin(ids)
    return df.loc[index, :].reset_index(drop=True)
    

def sample_by_id(df: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    unique_id = df['playerId'].unique()
    choosen_id = np.random.choice(unique_id, n, replace=False)
    return  filter_by_id(df, choosen_id)


def filter_by_date(df: pd.DataFrame,
                   date: str,
                   dt_col: str = 'date') -> pd.DataFrame:
    index = df.loc[:, dt_col] >= date
    return df.loc[index, :].reset_index(drop=True)
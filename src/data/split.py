from datetime import datetime
from typing import Union
import pandas as pd

class SplitData:
    """Helper class to split the data for time series"""
    def __init__(self, date: Union[datetime, str],
                 train_days: int = None,
                 test_days: int = 31,
                 gap: int = 0,
                 dt_col: str = 'date'):
        if isinstance(date, str):
            date = pd.to_datetime(date)
        # test range
        self.start_test_dt = date
        self.end_test_dt =  date + pd.to_timedelta(test_days, unit='d')
        # train range
        self.end_train_dt = date - pd.to_timedelta(gap, unit='d')
        self.start_train_dt = (self.end_train_dt - pd.to_timedelta(train_days, unit='d')
                               if train_days is not None else
                               None)
        self.dt_col = dt_col
    
    @staticmethod
    def _get_index(dates: pd.Series,
                start: datetime,
                end: datetime):
        index = (dates >= start) & (dates < end)
        return index
    
    def train_idx(self, df: pd.DataFrame) -> pd.DataFrame:
        start_train_dt = (df[self.dt_col].min()
                          if self.start_train_dt is None
                          else self.start_train_dt)
        index = self._get_index(df[self.dt_col], start_train_dt,
                                self.end_train_dt)
        return index

    def valid_idx(self, df: pd.DataFrame) -> pd.DataFrame:
        index = self._get_index(df[self.dt_col], self.start_test_dt,
                                self.end_test_dt)
        return index
    
    def train(self, df: pd.DataFrame):
        return self.filter(df, self.train_idx(df))
    
    def valid(self, df: pd.DataFrame):
        return self.filter(df, self.valid_idx(df))
    
    def filter(self, df: pd.DataFrame, index):
        return df.loc[index, :].reset_index(drop=True)
    
    def __repr__(self):
        return (f'test_range=({self.start_test_dt}, {self.end_test_dt}), '
                f'train_range=({self.start_train_dt}, {self.end_train_dt})')
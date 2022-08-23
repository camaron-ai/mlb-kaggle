from typing import Callable, Any
from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Callable
from sklearn.pipeline import Pipeline
import gc
import pandas as pd


@dataclass
class ModelOutput:
    model: Any
    predict_fn: Callable
    prediction: np.ndarray


def ensemble_pred(preds: List[np.ndarray]):
    return np.stack(preds, axis=1).mean(axis=1)

class Ensemble():
    def __init__(self, models: List[ModelOutput],
                 pipeline: Pipeline):
        self.models = models
        self.pipeline = pipeline
    
    def __call__(self, raw_df: pd.DataFrame):
        test_features = self.pipeline.transform(raw_df)
        return ensemble_pred([_model.predict_fn(test_features)
                              for _model in self.models])


def predict_recursive(test_df: pd.DataFrame,
                      raw_df: pd.DataFrame, 
                      predict_fn: Callable,
                      n_days: int,
                      target_cols: List[str]):
    
    assert test_df['date'].nunique() == 1, \
           'the test set has more than one date'

    test_date = test_df['date'].iloc[0]
    last_date = raw_df['date'].max()
    
    raw_df = raw_df[raw_df['date'] >= (last_date - pd.to_timedelta(n_days, unit='d'))]
    
    if test_date <= last_date:
        print('test date in training data')
        raw_df = raw_df[raw_df['date'] < test_date]
        last_date = raw_df['date'].max()
    assert test_date - last_date == pd.to_timedelta(1, unit='d'), \
        f'the test date ({test_date}) must be one day after the last_date ({last_date})'
    # append information
    raw_df = raw_df.append(test_df, ignore_index=True)
    # sort by index and dates
    raw_df.sort_values(by=['playerId', 'date'], inplace=True)
    raw_df.reset_index(drop=True, inplace=True)
    # get the index to locate the test df
    index = (raw_df['date'] == test_date)
    
    assert index.sum() == len(test_df)
    # predict
    prediction = predict_fn(raw_df)
    # filter prediction
    prediction = prediction[index]
    # create a dataframe with the prediction
    prediction_df = pd.DataFrame(prediction, columns=target_cols)
    prediction_df['date'] = test_date
    prediction_df['playerId'] = raw_df.loc[index, 'playerId'].to_numpy()
    print(prediction_df.head())
    # add the prediction to the dataset
    raw_df.loc[index, target_cols] = prediction
    
    del prediction, index
    gc.collect()
    return prediction_df, raw_df
    
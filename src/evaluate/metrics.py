from sklearn.metrics import mean_absolute_error
from typing import Callable
import numpy as np


def _calc_score(yhat: np.ndarray, target: np.ndarray,
                func: Callable, name: str = None):
    if name is None:
        name = func.__name__
    
    cw_error = [func(target[:, i], yhat[:, i])
                for i in range(target.shape[1])]    
    error = np.mean(cw_error)
    output = {f'{name}_mean': error}
    
    output.update({f'{name}_{i+1}': cw_error[i]
                   for i in range(len(cw_error))})
    return output


def compute_metrics(df, target_prefix: str = 'target',
                    yhat_prefix: str = 'yhat'):
    # make sure of numpy arrays
    yhat = df.loc[:, [yhat_prefix+str(i) for i in range(1, 5)]]
    target = df.loc[:, [target_prefix+str(i) for i in range(1, 5)]]

    return _calc_score(yhat.to_numpy(), target.to_numpy(),
                       func=mean_absolute_error, name='mae')
from lightgbm import LGBMRegressor
import pandas as pd
from typing import Dict, Any
import numpy as np
from train.core import ModelOutput ###no
import gc


def run_lgbm(config: Dict[str, Any],
             train_data: pd.DataFrame,
             valid_data:  pd.DataFrame, verbose=100):

    models = []
    train_features = train_data.loc[:, config.features]
    valid_features = valid_data.loc[:, config.features]

    for target_name in sorted(config.target_cols):
        print(target_name)
        _model = LGBMRegressor(**config.hp)
        _model.fit(train_features,
                   train_data.loc[:, target_name],
                   eval_set=[(valid_features, valid_data.loc[:, target_name])],  
                    early_stopping_rounds=verbose, 
                    verbose=verbose,
                    categorical_feature=config.categories)
        models.append( _model)

    def predict_fn(test_features: pd.DataFrame):
        return np.stack([_model.predict(test_features.loc[:, config.features])
                         for _model in models], axis=1)

    valid_prediction = predict_fn(valid_data)

    del train_features, valid_features
    gc.collect()
    return ModelOutput(models, predict_fn, valid_prediction)


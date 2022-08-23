import pandas as pd
from pipeline.core import make_unpack_tmf ###no
from typing import List, Dict

fields_type = Dict[str, pd.DataFrame] 

def unpack_dataframe(df: pd.DataFrame, fields: List[str]) -> fields_type:
    output = {}
    for field, output_name in fields.items():
        # check if there is data for this feature
        if df.loc[:, field].isna().all():
            output[output_name] = None
            continue
        # unpack data
        tmf = make_unpack_tmf(field)
        field_data = tmf.fit_transform(df)
        output[output_name] = field_data
    return output


def update_fields(train_fields: fields_type,
                  test_fields: fields_type,
                  add_field: bool = False,
                  concat: bool = True) -> fields_type:

    for field, test_data in test_fields.items():
        # if the field in the main dict, and if there is some data to add
        if (field in train_fields):
            train_data = train_fields[field]
            updated_data = (train_data.append(test_data, ignore_index=True)
                            if concat else test_data)
        elif not (field in train_fields) and add_field:
            updated_data = test_data
        train_fields[field] = updated_data
    return train_fields


def split_fields_by_date(fields: fields_type,
                 start_date: str = None,
                 end_date: str = None,
                 features: List[str] = None,
                 dt_col: str = 'date'):
    if features is None:
        features = list(fields.keys())

    for feature in features:
        data = fields[feature]
        if start_date is not None:
            index = (data[dt_col] >= start_date)
            if index.sum() == 0:
                print(feature, index.sum())
            else:
                data = data.loc[index, :]
        if end_date is not None:
            index = (data[dt_col] < start_date)
            assert index.sum() > 0
            data = data.loc[index, :]
        data.reset_index(drop=True, inplace=True)
        fields[feature] = data
    return fields

    
def has_duplicates(X: pd.DataFrame,
                   on: List[str] = ['playerId', 'date']) -> bool:
    return X.loc[:, on].duplicated().sum() > 0


def compute_rank_features(df: pd.DataFrame, on: List[str],
                  features: List[str]):
    suffix = '__' + '__'.join(on + ['ranked'])
    output_features = list(pd.Series(features) + suffix)
    ranked_features = df.groupby(on)[features].rank()
    df.loc[:, output_features] = ranked_features.fillna(0).to_numpy()
    return df


def normalize_with_max(df: pd.DataFrame,
                       on: List[str],
                       features: List[str]):
    maximum = df.groupby(on)[features].transform('max')
    output_features = [f'__'.join([f] + on + ['maxNorm'])
                       for f in features]
    normalized_features = df.loc[:, features] / maximum.to_numpy()
    df.loc[:, output_features] = normalized_features.to_numpy()
    
    return df


def add_suffix(df: pd.DataFrame, features: List[str],
               suffix: str):
    new_features_names = {name: f'{name}__{suffix}'
                          for name in features}
    return df.rename(columns=new_features_names)
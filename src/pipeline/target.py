import pandas as pd


def target_preprocessing(target: pd.DataFrame) -> pd.DataFrame:
    target.drop('engagementMetricsDate', axis=1, inplace=True)
    target.sort_values(by=['playerId', 'date'], inplace=True)
    target.reset_index(drop=True, inplace=True)
    return target
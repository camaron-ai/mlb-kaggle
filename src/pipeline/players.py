import numpy as np
import pandas as pd


def player_preprocessing(players: pd.DataFrame) -> pd.DataFrame:    
    # weight to kg and height to cm
    players['weight'] *= 0.453592
    players['height'] = players['heightInches'] * 2.54 / 100
    players['playerBMI'] = players['weight'] / np.power(players['height'], 2)
    
    # drop unnecessary features
    to_drop = ['birthCity', 'heightInches',
               'birthStateProvince',
               'primaryPositionCode',
               'playerName',
               'playerForTestSetAndFuturePreds']
    players.drop(to_drop, inplace=True, axis=1)
    return players


def join_players_info(df: pd.DataFrame, path_to_players_csv: str):
    # read players csv
    raw_players = pd.read_csv(path_to_players_csv, parse_dates=['DOB', 'mlbDebutDate'])
    # process player data
    players = player_preprocessing(raw_players)
    # merge
    return df.merge(players, how='left', on=['playerId'])

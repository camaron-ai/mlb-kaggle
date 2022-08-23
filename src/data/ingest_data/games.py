import pandas as pd
import numpy as np


def preprocess_games_stats(games: pd.DataFrame):    
    to_drop = ['gameTimeUTC', 'resumeDate',
               'resumedFrom', 'codedGameState',
               'detailedGameState',
               'gameNumber',
               'doubleHeader',
               'dayNight',
               'scheduledInnings', 
               'homeName',
               'homeAbbrev',
               'gameType',
               'homeWins',
               'homeLosses',
               'homeWinPct',
               'awayWins',
               'awayLosses',
               'awayWinPct',
               'homeName',
               'gameDate',
               'awayName', 'awayAbbrev', 'isTie']
    games['gamesInSeries'] = (games['gamesInSeries'].replace({0: 1})
                              .fillna(1).astype(np.int64))
    # drop features
    
    games.sort_values(by=['date', 'gamePk'], inplace=True)
    games.reset_index(drop=True, inplace=True)
    return games.drop(to_drop, axis=1)


def _join_home_and_away_games(games: pd.DataFrame):
    # home games
    home_team_games = games.copy()
    home_team_games['home'] = 1
    home_team_games['away'] = 0
    home_team_games.rename(columns={'homeId': 'teamA',
                                    'awayId': 'teamB'}, inplace=True)

    # away games
    away_team_games = games.copy()
    away_team_games['home'] = 0
    away_team_games['away'] = 1
    away_team_games.rename(columns={'homeId': 'teamB',
                                    'awayId': 'teamA'}, inplace=True)
    
    team_games = pd.concat([home_team_games, away_team_games],
                               axis=0, ignore_index=True)
    
    # sort values
    team_games = (team_games.sort_values(by=['teamA', 'teamB', 'date'])
                  .reset_index(drop=True))
    
    return team_games


def compute_current_game_in_series(games: pd.DataFrame):
    team_games = _join_home_and_away_games(games)
    
    features = ['teamA', 'teamB', 'date', 'season', 'seriesDescription']
    team_games = team_games.loc[:, features]
    
    team_games['currentGameInSeries'] = 1

    current_series_game = (team_games.set_index('date')
                           .groupby(['teamA', 'teamB', 'season',
                                     'seriesDescription'])['currentGameInSeries']
                           .expanding().sum())

    current_series_game = current_series_game.reset_index()
    current_series_game.drop_duplicates(subset=['teamA', 'teamB',
                                                'date'], keep='last', inplace=True)
    current_series_game.reset_index(drop=True, inplace=True)    
    return current_series_game


def compute_current_game_in_series_and_join(games: pd.DataFrame):
    current_series_game = compute_current_game_in_series(games)
    current_series_game.drop(['season', 'seriesDescription'], axis=1, inplace=True)
    games = games.merge(current_series_game,
                        left_on=['homeId', 'awayId', 'date'],
                        right_on=['teamA', 'teamB', 'date'],
                        how='left')
    
    games.drop(['teamA', 'teamB'], axis=1, inplace=True)
    games['currentGameInSeries'] = ((games['currentGameInSeries'] - 1) %
                                     games['gamesInSeries'])
    return games
    

def compute_games_stats(games: pd.DataFrame) -> pd.DataFrame:
    team_games = _join_home_and_away_games(games)
    team_games = team_games.loc[:, ['teamA', 'teamB', 'date', 'season',
                                    'awayWinner', 'homeWinner', 'home', 'away']]
    # fill the nan with False
    # this is because you cant win at home when playing as visitant
    team_games['homeWinner'] = team_games['homeWinner'] * team_games['home']
    team_games['awayWinner'] = team_games['awayWinner'] * team_games['away']
    # some day have different days, lets sum over each day
    team_games = team_games.groupby(['teamA', 'teamB',  'season', 'date']).sum()
    team_games.reset_index(inplace=True)
    
    # sort values
    team_games = (team_games.sort_values(by=['teamA', 'teamB', 'date'])
                  .reset_index(drop=True))
    # calculate the cummulative sum
    team_games = (team_games.set_index('date')
                  .groupby(['teamA', 'teamB', 'season'])[['home', 'away', 'homeWinner', 'awayWinner']]
                  .expanding().sum())
    # compute stats
    team_games['totalGamesVsoppTeam'] = (team_games['away'] + team_games['home'])
    team_games['WinPctAsHome'] = team_games['homeWinner'] / team_games['home']
    team_games['WinPctAsAway'] = team_games['awayWinner'] / team_games['away']
    team_games['WintPctHist'] = ((team_games['homeWinner'] + team_games['awayWinner'])
                                 /  team_games['totalGamesVsoppTeam'])
    # fillnan with 0
    team_games.replace([np.inf, -np.inf], np.nan, inplace=True)
    team_games.fillna(0, inplace=True)
    # reset the index
    team_games.reset_index(inplace=True)
    team_games.drop(['away', 'home', 'homeWinner', 'awayWinner', 'season'], axis=1, inplace=True)
    return team_games


def compute_games_stats_and_join(games: pd.DataFrame) -> pd.DataFrame:
    input_shape = len(games)
    team_games = compute_games_stats(games)
    features = ['WinPctAsHome', 'WinPctAsAway', 'WintPctHist']
    # merge for the home ids
    games = games.merge(team_games, left_on=['homeId', 'awayId', 'date'],
            right_on=['teamA', 'teamB', 'date'],
            how='left')
    # rename columns to start with home
    games.drop(['teamA', 'teamB', 'totalGamesVsoppTeam'], inplace=True, axis=1)
    games.rename(columns={f: 'home' + f for f in features}, inplace=True)
    
    # merge for the away teams
    games = games.merge(team_games, left_on=['homeId', 'awayId', 'date'],
                right_on=['teamB', 'teamA', 'date'],
                how='left')
    # rename columns to start with waway
    games.rename(columns={f: 'away' + f for f in features}, inplace=True)
    games.drop(['teamA', 'teamB'], inplace=True, axis=1)
    
    assert len(games) == input_shape, \
           f'the input lenght (input_shape) != output shape (len(games))'
    return games


def ingest_games_stats(games: pd.DataFrame):
    games = preprocess_games_stats(games)
    games = compute_current_game_in_series_and_join(games)
    games = compute_games_stats_and_join(games)
    games.drop(['season', 'gamesInSeries', 'seriesDescription'], axis=1, inplace=True)
    return games

def join_games_stats_to_pstats(pstats: pd.DataFrame,
                               games: pd.DataFrame):
    teamFeatures = ['Id',
                    'Winner',
                    'Score',
                    'WinPctAsHome',
                    'WinPctAsAway',
                    'WintPctHist']

    add_suffix = lambda suffix: [suffix + f for f in teamFeatures] 

    homeFeatures = add_suffix('home')
    awayFeatures = add_suffix('away')
    playerTeamFeatures = add_suffix('playerTeam')
    opTeamFeatures = add_suffix('opponentTeam')
    
    pstats = pstats.merge(games, on=['gamePk', 'date'], how='left')
    # home and aways stats to player and opponent stats
    pstats.loc[:, playerTeamFeatures] = np.where(pstats[['home']], pstats[homeFeatures], pstats[awayFeatures])
    pstats.loc[:, opTeamFeatures] = np.where(pstats[['home']], pstats[awayFeatures], pstats[homeFeatures])
    # drop features
    redundat_features = ['playerTeamId', 'opponentTeamWinner', 'opponentTeamWintPctHist']
    pstats.drop((homeFeatures + awayFeatures +
                 redundat_features),
                axis=1, inplace=True)
    # compute the difference between the scores
    return pstats
import pandas as pd
import numpy as np
from typing import List
from data.ingest_data.core import compute_rank_features ###no


def preprocess_standings(standings: pd.DataFrame):
    def _streak_code(standings):
        standings['streakCode'] = standings['streakCode'].fillna('W0')
        streak = standings['streakCode'].str[1:].astype(np.float32)
        code = np.where(standings['streakCode'].str[:1] == 'W', 1, -1)
        standings['streak'] = streak * code
        standings.drop('streakCode', axis=1, inplace=True)
    to_drop = ['gameDate', 'teamName',
               'leagueGamesBack',
               'sportGamesBack', 'divisionGamesBack', 
               'runsAllowed', 'runsScored', 
               'extraInningWins', 'extraInningLosses', 
               'oneRunWins',
               'oneRunLosses', 'dayWins', 'dayLosses',
               'nightWins', 'nightLosses',
               'grassWins', 'grassLosses', 'turfWins',
               'turfLosses', 'divWins',
               'divLosses', 'alWins', 'alLosses',
               'nlWins', 'nlLosses', 'season', 
               'wins', 'losses', 'wildCardEliminationNumber',
               'eliminationNumber', 'divisionRank', 'leagueRank', 'wildCardRank',
               'divisionId', 'streakCode']
    standings = standings.drop(to_drop, axis=1)
    # _streak_code(standings)
    standings['wildCardLeader'] = (standings['wildCardLeader'].replace({'None': False})
                                   .fillna(False).astype(np.bool_))

    bool_features = ['divisionChamp', 'divisionLeader']
    standings[bool_features] = standings[bool_features].astype('float')

    standings['homeWinPct'] = standings['homeWins'] / (standings['homeWins'] + standings['homeLosses'])
    standings['awayWinPct'] = standings['awayWins'] / (standings['awayWins'] + standings['awayLosses'])

    standings.rename(columns={'pct': 'winPct'}, inplace=True)
    standings = standings.drop(['homeWins', 'homeLosses',
                                 'awayWins', 'awayLosses'], axis=1)
    # scale up to 1
    standings[['lastTenWins', 'lastTenLosses']] /= 10
    return standings
    

def compute_standings_features(standings: pd.DataFrame, features: List[str]):
    return compute_rank_features(standings, on=['date'], features=features)


def ingest_standings(standings: pd.DataFrame):
    standings = preprocess_standings(standings)
    standings = compute_standings_features(standings, features=['homeWinPct', 'awayWinPct', 'winPct'])
    return standings


# def join_standings_to(df: pd.DataFrame,
#                       standings: pd.DataFrame,
#                       as_away: bool = False):
#     team_col = 'teamId' if as_away else 'opponentTeamId'
#     output_name = 'playerTeam' if as_away else 'opponentTeam'
#     _standings = standings.rename(columns={'teamId': team_col})

#     df = df.merge(_standings, how='left', on=['date', team_col])
#     df.rename(columns={f: f'{output_name}{f.title()}St'
#                            for f in _standings.columns.drop([team_col, 'date'])}, inplace=True)
#     return df


def join_standings_to(df: pd.DataFrame,
                      standings: pd.DataFrame):
    def _merge(df: pd.DataFrame, standings, team_col: str, output_name: str):
        df = df.merge(standings, how='left', on=['date', team_col])
        df.rename(columns={f: f'{output_name}{f.title()}'
                            for f in standings.columns.drop([team_col, 'date'])}, inplace=True)
        return df

    df = _merge(df, standings, team_col='teamId',
                output_name='playerTeam')
    # if 'opponentTeamId' not in df.columns:
        # return df
    # standings = standings.rename(columns={'teamId': 'opponentTeamId'})
    # df = _merge(df, standings, team_col='opponentTeamId',
                # output_name='opponentTeam')
    return df

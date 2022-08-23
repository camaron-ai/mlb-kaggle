from data.ingest_data.core import has_duplicates, add_suffix ###no
from data.ingest_data.core import compute_rank_features, normalize_with_max ###no
import pandas as pd
from typing import List

team_stats_features = ['runsScored', 'homeRuns', 'strikeOuts', 'hits', 'runsPitching',
                       'homeRunsPitching', 'outsPitching','rbiPitching']

rank_features = ['runsScored', 'homeRuns']

def preprocess_teams(teams: pd.DataFrame):
    to_keep = ['teamId', 'date']
    # we drop teamId because this info is in roster]
    agg_teams = (teams.groupby(['teamId', 'date'])[team_stats_features]
                  .sum().reset_index())

    teams = (teams.drop_duplicates(subset=['teamId', 'date'], keep='last')
              .loc[:, to_keep])
    
    teams = teams.merge(agg_teams, on=['teamId', 'date'], how='left')
    assert not has_duplicates(teams, on=['teamId', 'date']), 'team stats include duplicates'
    return teams

def ingest_team_stats(teams: pd.DataFrame):
    teams = preprocess_teams(teams)
    teams = normalize_with_max(teams, on=['date'],
                                features=rank_features)
    return teams

def join_team_stats_to_pstats(pstats: pd.DataFrame,
                              teams: pd.DataFrame):
    features = teams.columns.drop(['date', 'teamId'])
    player_features = list('playerTeam' + features)
    away_features = list('opponentTeam' + features)
    
    # for the player team
    home_teams = teams.rename(columns={old: new for old, new in zip(features, player_features)})
    pstats = pstats.merge(home_teams, on=['teamId', 'date'], how='left')
    
    # for the away team
    teams.rename(columns={'teamId': 'opponentTeamId'}, inplace=True)
    away_teams = teams.rename(columns={old: new for old, new in zip(features, away_features)})
    pstats = pstats.merge(away_teams, on=['opponentTeamId', 'date'], how='left')
    return pstats

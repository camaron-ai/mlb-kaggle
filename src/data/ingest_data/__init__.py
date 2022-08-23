from pipeline.target import target_preprocessing ###no
from pipeline.players import join_players_info ###no
from pipeline.season import join_season_info ###no
from data.ingest_data.player_stats import ingest_player_stats ###no
from data.ingest_data.standings import ingest_standings, join_standings_to ###no
from data.ingest_data.games import ingest_games_stats, join_games_stats_to_pstats ###no
from data.ingest_data.core import has_duplicates, unpack_dataframe, update_fields, normalize_with_max ###no
from data.ingest_data.team_stats import ingest_team_stats, join_team_stats_to_pstats ###no
from data.ingest_data.events import ingest_events ###no
import pandas as pd
from pathlib import Path
from typing import List, Dict
import os
import numpy as np
import gc


feature_fields = {'games': 'games', 
                   'playerBoxScores': 'pstats',
                   'rosters': 'rosters',
                   'playerTwitterFollowers': 'ptw_fl',
                   'teamTwitterFollowers': 'team_tw_fl', 
                   'awards': 'awards',
                   'teamBoxScores': 'teams',
                #    'transactions': 'transactions',
                   'events': 'events',
                   'standings': 'standings'}


def ingest_target(df: pd.DataFrame) -> pd.DataFrame:
    print('preprocessing target')
    fields = unpack_dataframe(df, fields={'nextDayPlayerEngagement': 'target'})
    target = fields['target']
    return target_preprocessing(target)


def ingest_stats_features(pstats: pd.DataFrame,
                          games: pd.DataFrame,
                          teams: pd.DataFrame, 
                          events: pd.DataFrame):
    teams = ingest_team_stats(teams)
    games = ingest_games_stats(games)
    events = ingest_events(events)
    pstats = ingest_player_stats(pstats)
    pstats = join_games_stats_to_pstats(pstats, games)
    pstats = join_team_stats_to_pstats(pstats, teams)
    pstats = pstats.merge(events, on=['playerId', 'date'], how='left')
    pstats.drop(['gamePk', 'teamId'], axis=1, inplace=True)
    return pstats


def ingest_rosters(rosters: pd.DataFrame):
    # rename roster date
    rosters.rename(columns={'gameDate': 'rosterDate'}, inplace=True)
    # drop statusCode
    rosters.drop(['statusCode'], axis=1, inplace=True)
    assert not has_duplicates(rosters), 'rosters include duplicates'
    return rosters


def ingest_player_twitter_fl(tw_fl: pd.DataFrame):
    tw_fl = tw_fl.loc[:, ['playerId', 'numberOfFollowers', 'date']]
    assert not has_duplicates(tw_fl), 'player tw include duplicates'
    return tw_fl


def ingest_team_twitter_fl(tw_fl: pd.DataFrame):
    tw_fl.rename(columns={'numberOfFollowers': 'teamFollowers'}, inplace=True)
    tw_fl = tw_fl.loc[:, ['teamId', 'teamFollowers', 'date']]
    assert not has_duplicates(tw_fl, on=['date', 'teamId']), 'team tw include duplicates'
    return tw_fl


def ingest_awards(awards: pd.DataFrame):
    awards = awards[['date', 'awardId', 'playerId']]
    awards = awards.groupby(['date', 'playerId'])[['awardId']].count()
    awards.rename(columns={'awardId': 'awardCount'}, inplace=True)
    awards.reset_index(inplace=True)
    # awards['totalAwardCount'] = awards.groupby(['playerId'])[['awardId']].count()
    assert not has_duplicates(awards), 'awards tw include duplicates'
    return awards


def ingest_transactions(transactions: pd.DataFrame):
    transactions = transactions.dropna(subset=['playerId'])
    transactions.loc[:, 'playerId'] = transactions.loc[:, 'playerId'].astype(np.int64)
    transactions = transactions.loc[:, ['playerId', 'toTeamId', 'date']]
    transactions.rename(columns={'toTeamId': 'MoveToTeamId'}, inplace=True)
    transactions.drop_duplicates(subset=['date', 'playerId'], keep='last', inplace=True)
    transactions.reset_index(drop=True, inplace=True)
    assert not has_duplicates(transactions), 'transactions has duplicates'
    return transactions


def ingest_features(df: pd.DataFrame,
                    pstats: pd.DataFrame = None,
                    games: pd.DataFrame = None,
                    awards: pd.DataFrame = None,
                    ptw_fl: pd.DataFrame = None,
                    rosters: pd.DataFrame = None,
                    standings: pd.DataFrame = None,
                    team_tw_fl: pd.DataFrame = None,
                    teams: pd.DataFrame = None,
                    events: pd.DataFrame=None,
                    transactions: pd.DataFrame = None,
                    path_to_players_csv: str = None,
                    path_to_season_csv: str = None):


    if standings is not None:
        standings = ingest_standings(standings)
    if (pstats is not None) and (games is not None):
        pstats = ingest_stats_features(pstats, games, teams, events)
    if rosters is not None:
        rosters = ingest_rosters(rosters)
    if (awards is not None):
        awards = ingest_awards(awards)
    if ptw_fl is not None:
        ptw_fl = ingest_player_twitter_fl(ptw_fl)
    if team_tw_fl is not None:
        team_tw_fl = ingest_team_twitter_fl(team_tw_fl)
    if transactions is not None:
        transactions = ingest_transactions(transactions)
    
    for feature_ds in [pstats, rosters, ptw_fl, awards, transactions]:
        if feature_ds is None:
            continue
        df = df.merge(feature_ds, on=['playerId', 'date'],
                      how='left')

    if (team_tw_fl is not None) and ('teamId' in df.columns):
        df = df.merge(team_tw_fl, on=['teamId', 'date'],
                    how='left')

    if (standings is not None) and ('teamId' in df.columns):
        df = join_standings_to(df, standings)

    if path_to_season_csv is not None:
        df = join_season_info(df, path_to_season_csv=path_to_season_csv)
    if path_to_players_csv is not None:
        df = join_players_info(df, path_to_players_csv=path_to_players_csv)
    assert not has_duplicates(df), 'output features include duplicates'
    return df


def create_test_template(sample_submission: pd.DataFrame):
    test_df = sample_submission.copy(deep=True)
    test_df.reset_index(inplace=True)
    test_df['date'] = pd.to_datetime(test_df['date'], format='%Y%m%d')
    
    test_df.sort_values(by=['playerId', 'date'], inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    return test_df.drop('date_playerId', axis=1)


def ingest_features_for_test(test_df: pd.DataFrame,
                             raw_test_df: pd.DataFrame,
                             train_fields: Dict[str, pd.DataFrame], 
                             path_to_players_csv: str = None,
                             path_to_season_csv: str = None):
    # read each feature dataset
    test_fields = unpack_dataframe(raw_test_df, fields=feature_fields)
    # update the information in the train fields
    test_fields = update_fields(train_fields, test_fields)
    # compute all features
    test_df = ingest_features(test_df, **test_fields,
                              path_to_players_csv=path_to_players_csv,
                              path_to_season_csv=path_to_season_csv)
    return test_df, test_fields


def ingest_test_data(submission_template: pd.DataFrame,
                     raw_test_df: pd.DataFrame,
                     train_fields: Dict[str, pd.DataFrame],
                     path_to_players_csv: str = None,
                     path_to_season_csv: str = None):
    submission_template['playerId'] = (submission_template['date_playerId']
                                      .map(lambda x: int(x.split('_')[1])))
    test_df = create_test_template(submission_template)

    test_date = test_df['date'].iloc[0]
    raw_test_df['date'] = test_date
    raw_test_df.set_index('date', inplace=True)

    test_df, test_fields = ingest_features_for_test(test_df, raw_test_df,
                                       train_fields,
                                       path_to_players_csv=path_to_players_csv,
                                       path_to_season_csv=path_to_season_csv) 
    return submission_template, test_df, test_fields


def ingest_train_data(path_to_train_csv: str,
                      path_to_players_csv: str,
                      path_to_season_csv: str) -> pd.DataFrame:
    print('reading training data..')
    train_data = pd.read_csv(path_to_train_csv, parse_dates=['date'])
    # set index date
    train_data = train_data.set_index('date')
    # ingest target data
    df = ingest_target(train_data)

    feature_fields_data = unpack_dataframe(train_data, fields=feature_fields)
    del train_data
    gc.collect()
    df = ingest_features(df, **feature_fields_data,
                        path_to_players_csv=path_to_players_csv,
                        path_to_season_csv=path_to_season_csv)
    return df



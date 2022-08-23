from data.ingest_data.core import has_duplicates, add_suffix ###no
from data.ingest_data.core import compute_rank_features, normalize_with_max ###no
import pandas as pd
from typing import List

player_stats_features = ['battingOrder', 'gamesPlayedBatting', 'flyOuts',
       'groundOuts', 'runsScored', 'doubles', 'triples', 'homeRuns',
       'strikeOuts', 'baseOnBalls', 'intentionalWalks', 'hits', 'hitByPitch',
       'atBats', 'caughtStealing', 'stolenBases', 'groundIntoDoublePlay',
       'groundIntoTriplePlay', 'plateAppearances', 'totalBases', 'rbi',
       'leftOnBase', 'sacBunts', 'sacFlies', 'catchersInterference',
       'pickoffs', 'gamesPlayedPitching', 'gamesStartedPitching',
       'completeGamesPitching', 'shutoutsPitching', 'winsPitching',
       'lossesPitching', 'flyOutsPitching', 'airOutsPitching',
       'groundOutsPitching', 'runsPitching', 'doublesPitching',
       'triplesPitching', 'homeRunsPitching', 'strikeOutsPitching',
       'baseOnBallsPitching', 'intentionalWalksPitching', 'hitsPitching',
       'hitByPitchPitching', 'atBatsPitching', 'caughtStealingPitching',
       'stolenBasesPitching', 'inningsPitched', 'saveOpportunities',
       'earnedRuns', 'battersFaced', 'outsPitching', 'pitchesThrown', 'balls',
       'strikes', 'hitBatsmen', 'balks', 'wildPitches', 'pickoffsPitching',
       'rbiPitching', 'gamesFinishedPitching', 'inheritedRunners',
       'inheritedRunnersScored', 'catchersInterferencePitching',
       'sacBuntsPitching', 'sacFliesPitching', 'saves', 'holds', 'blownSaves',
       'assists', 'putOuts', 'errors', 'chances']


players_features_to_drop = ['gamesPlayedBatting', 'flyOuts', 'doubles', 'triples', 'atBats',
       'caughtStealing', 'groundIntoDoublePlay', 'leftOnBase', 'sacBunts',
       'sacFlies', 'shutoutsPitching', 'flyOutsPitching',
       'airOutsPitching', 'doublesPitching',
       'triplesPitching', 'homeRunsPitching', 'baseOnBallsPitching',
       'intentionalWalksPitching', 'hitsPitching', 'hitByPitchPitching',
       'stolenBasesPitching', 'earnedRuns', 'pitchesThrown', 'balls',
       'strikes', 'hitBatsmen', 'wildPitches', 'rbiPitching',
       'inheritedRunnersScored', 'sacFliesPitching', 'errors']


rank_features = ['runsScored', 'homeRuns', 'hits', 'SLG', 'rbi', 'runsPitching']

def compute_player_metrics(pstats: pd.DataFrame):
    pstats['SLG'] = ((1 * pstats['hits'] +
                      2 * pstats['doubles'] +
                      3 * pstats['triples'] +
                      4 * pstats['homeRuns']) / pstats['atBats']).fillna(-1).to_numpy()
    return pstats
    
to_keep = ['home', 'gamePk', 'playerId', 'date', 'teamId', 'positionName']

def preprocess_player_stats(pstats: pd.DataFrame):
    # we drop teamId because this info is in roster]
    agg_pstats = (pstats.groupby(['playerId', 'date'])[player_stats_features]
                  .sum().reset_index())

    pstats = (pstats.drop_duplicates(subset=['playerId', 'date'], keep='last')
              .loc[:, to_keep])
    
    pstats['pstatsDate'] = pstats['date'].copy(deep=True)
    pstats = pstats.merge(agg_pstats, on=['playerId', 'date'], how='left')
    assert not has_duplicates(pstats), 'player stats include duplicates'
    return pstats


def ingest_player_stats(pstats: pd.DataFrame):
    pstats = preprocess_player_stats(pstats)
    pstats = compute_player_metrics(pstats)
    # rank features per game
    # pstats = compute_rank_features(pstats, on=['gamePk'],
                                #    features=rank_features)
    pstats = normalize_with_max(pstats, on=['date'],
                                features=rank_features)
    pstats = pstats.drop(players_features_to_drop, axis=1)
    # stats_features = pstats.columns.drop(to_keep + ['pstatsDate'])
    # pstats = add_suffix(pstats, features=stats_features)
    return pstats
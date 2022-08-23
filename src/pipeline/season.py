from typing import List
import pandas as pd
import numpy as np


def date_preprocessing(df: pd.DataFrame,
                       path_to_season: str,
                       dt_col: str = 'date',
                       date_attr: List[str] = ['year', 'month']):
    assert 'year' in date_attr, \
            'year attr must be on the date_attr list'
    date_cols = ['seasonStartDate', 'seasonEndDate', 'preSeasonStartDate',
                 'preSeasonEndDate', 'regularSeasonStartDate', 'regularSeasonEndDate',
                 'lastDate1stHalf', 'allStarDate', 'firstDate2ndHalf',
                 'postSeasonStartDate', 'postSeasonEndDate']
    seasons = pd.read_csv(path_to_season, parse_dates=date_cols)
    
    # adding date attr
    for attr in date_attr:
        attr = attr.lower()
        df[attr] = getattr(df[dt_col].dt, attr)

    season_df = pd.merge(df, seasons, left_on='year', right_on='seasonId')

    season_df['inSeason'] = (season_df['date'].between(
                                    season_df['regularSeasonStartDate'],
                                    season_df['postSeasonEndDate'],
                                    inclusive = True
                                    )
                                  )

    season_df['seasonPart'] = np.select(
      [
        season_df['date'] < season_df['preSeasonStartDate'], 
        season_df['date'] < season_df['regularSeasonStartDate'],
        season_df['date'] <= season_df['lastDate1stHalf'],
        season_df['date'] < season_df['firstDate2ndHalf'],
        season_df['date'] <= season_df['regularSeasonEndDate'],
        season_df['date'] < season_df['postSeasonStartDate'],
        season_df['date'] <= season_df['postSeasonEndDate'],
        season_df['date'] > season_df['postSeasonEndDate']
      ], 
      [
        'Offseason',
        'Preseason',
        'Reg Season 1st Half',
        'All-Star Break',
        'Reg Season 2nd Half',
        'Between Reg and Postseason',
        'Postseason',
        'Offseason'
      ], 
      default = np.nan
      )

    season_df.drop(seasons.columns, axis=1, inplace=True)
    
    return season_df
  

def join_season_info(df: pd.DataFrame,
                    path_to_season_csv: str,
                    date_attr: List[str] = ['year', 'weekday'],
                    ):
    # get unique dates
    dates = df[['date']].drop_duplicates()
    # add season info
    dates = date_preprocessing(dates, path_to_season_csv,
                   date_attr=date_attr)
    # merge
    return df.merge(dates, how='left', on=['date'])
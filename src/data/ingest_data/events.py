import pandas as pd
import numpy as np
import gc


def ingest_events(events: pd.DataFrame):
    def concat_text(x):
        return ' EndEvent '.join(x)

    to_keep = ['date', 'hitterId', 'pitcherId', 'description']
    to_drop = [f for f in events.columns 
               if f not in to_keep]
    events.drop(to_drop, inplace=True, axis=1)
    gc.collect()
    events = events.dropna(subset=['description'])
    
    hitter_events = events.groupby(['date', 'hitterId'])['description'].apply(concat_text)
    pitcher_events = events.groupby(['date', 'pitcherId'])['description'].apply(concat_text)
    
    hitter_events = hitter_events.reset_index()
    pitcher_events = pitcher_events.reset_index()
    
    hitter_events.rename(columns={'hitterId': 'playerId'}, inplace=True)
    pitcher_events.rename(columns={'pitcherId': 'playerId'}, inplace=True)
    
    text_events = pd.concat([hitter_events, pitcher_events], ignore_index=True)
    text_events = text_events.groupby(['date', 'playerId'])['description'].apply(concat_text)
    
    return text_events.reset_index()
    
    
    
import pandas as pd

def func(x):
    cr = x['Capture_rate']
    if 0 <= cr <= 0.25: return 0
    elif 0.25 < cr <= 0.5: return 1
    elif 0.5 < cr <= 0.75: return 2
    elif 0.75 < cr <= 1: return 3
    else: return None

def discretize(df):
    
    df['Capture_rate'] = df.apply(func, axis=1)
    df['Capture_rate'] = df['Capture_rate'].astype('int64')
    return df


import pandas as pd, numpy as np

def compute_rsi(series, window=14):
    delta=series.diff().dropna()
    up=delta.clip(lower=0); down=-1*delta.clip(upper=0)
    roll_up=up.rolling(window).mean(); roll_down=down.rolling(window).mean()
    rs=roll_up/(roll_down+1e-9)
    return 100-(100/(1+rs))

def make_features(df,horizon=3):
    df=df.copy().sort_index()
    df['ret1']=df['Close'].pct_change(1)
    df['ret5']=df['Close'].pct_change(5)
    df['ma20']=df['Close'].rolling(20).mean()
    df['ema12']=df['Close'].ewm(span=12,adjust=False).mean()
    df['ema26']=df['Close'].ewm(span=26,adjust=False).mean()
    df['macd']=df['ema12']-df['ema26']
    df['rsi']=compute_rsi(df['Close'],14)
    df['vol20']=df['Close'].pct_change().rolling(20).std()
    df['future_close']=df['Close'].shift(-horizon)
    df['target']=(df['future_close']>df['Close']).astype(int)
    df=df.dropna()
    X=df[['ret1','ret5','ma20','ema12','ema26','macd','rsi','vol20']]
    y=df['target']
    return X,y,df

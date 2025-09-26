import os, time, logging
from datetime import datetime
import pandas as pd
try:
    import yfinance as yf
    _HAS_YFINANCE=True
except:
    _HAS_YFINANCE=False

def normalize_ticker_for_yf(ticker: str):
    if '.' in ticker: return ticker.upper()
    return ticker.upper()+'.NS'

def fetch_current_price(user_ticker: str):
    if not _HAS_YFINANCE:
        return {'price': None,'ok':False,'source':'none','ts':datetime.utcnow()}
    try:
        t=yf.Ticker(normalize_ticker_for_yf(user_ticker))
        p=None
        if hasattr(t,'fast_info'):
            p=t.fast_info.get('last_price')
        if not p:
            p=t.info.get('regularMarketPrice')
        if not p:
            hist=t.history(period='5d')
            if not hist.empty: p=float(hist['Close'].iloc[-1])
        if p:
            return {'price':float(p),'ok':True,'source':'yfinance','ts':datetime.utcnow()}
    except Exception as e:
        return {'price':None,'ok':False,'error':str(e),'source':'yfinance','ts':datetime.utcnow()}
    return {'price':None,'ok':False,'source':'none','ts':datetime.utcnow()}

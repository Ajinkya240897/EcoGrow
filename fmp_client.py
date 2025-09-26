import requests
BASE='https://financialmodelingprep.com/api/v3'
def safe_fetch(symbol, api_key):
    profile,metrics=None,None
    try:
        r=requests.get(f"{BASE}/profile/{symbol}?apikey={api_key}",timeout=6)
        if r.ok:
            d=r.json()
            if isinstance(d,list) and d: profile=d[0]
    except: pass
    try:
        r=requests.get(f"{BASE}/ratios/{symbol}?limit=1&apikey={api_key}",timeout=6)
        if r.ok:
            d=r.json()
            if isinstance(d,list) and d: metrics=d[0]
    except: pass
    return {'profile':profile,'metrics':metrics}

import os,joblib,numpy as np,yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from ml.feature_engineering import make_features

try:
    from catboost import CatBoostClassifier
    _HAS_CAT=True
except:
    _HAS_CAT=False

def train_stack(ticker,horizon,out):
    df=yf.download(ticker+'.NS',period='4y')
    X,y,_=make_features(df,horizon=horizon)
    tscv=TimeSeriesSplit(n_splits=5)
    meta_X=[];meta_y=[]
    for tr,te in tscv.split(X):
        Xtr,Xte=X.iloc[tr],X.iloc[te];ytr,yte=y.iloc[tr],y.iloc[te]
        l=lgb.LGBMClassifier(n_estimators=300).fit(Xtr,ytr)
        p_l=l.predict_proba(Xte)[:,1]
        if _HAS_CAT:
            c=CatBoostClassifier(iterations=300,verbose=0).fit(Xtr,ytr)
            p_c=c.predict_proba(Xte)[:,1]
        else:
            p_c=np.zeros_like(p_l)
        meta_X.append(np.vstack([p_l,p_c]).T);meta_y.append(yte.values)
    MX=np.vstack(meta_X);My=np.concatenate(meta_y)
    stacker=LogisticRegression().fit(MX,My)
    os.makedirs(os.path.dirname(out),exist_ok=True)
    joblib.dump({'stacker':stacker},out)

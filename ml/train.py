import argparse,os,joblib,yfinance as yf
from sklearn.model_selection import TimeSeriesSplit,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from ml.feature_engineering import make_features

def train_model(ticker,horizon,out):
    df=yf.download(ticker+'.NS',period='4y')
    X,y,_=make_features(df,horizon=horizon)
    clf=lgb.LGBMClassifier(n_estimators=400,learning_rate=0.05)
    pipe=Pipeline([('scaler',StandardScaler()),('clf',clf)])
    tscv=TimeSeriesSplit(n_splits=5)
    scores=cross_val_score(pipe,X,y,cv=tscv,scoring='accuracy')
    print('CV mean:',scores.mean())
    pipe.fit(X,y)
    os.makedirs(os.path.dirname(out),exist_ok=True)
    joblib.dump(pipe,out)
    print('Saved to',out)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--ticker',required=True)
    ap.add_argument('--horizon',type=int,default=3)
    ap.add_argument('--out',default='models/model_lgb.pkl')
    a=ap.parse_args()
    train_model(a.ticker,a.horizon,a.out)

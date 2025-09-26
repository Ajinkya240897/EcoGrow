import argparse,os,joblib,yfinance as yf,optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit,cross_val_score
import lightgbm as lgb
from ml.feature_engineering import make_features

def objective(trial,X,y):
    params={'n_estimators':trial.suggest_int('n_estimators',100,800),
            'learning_rate':trial.suggest_float('learning_rate',0.01,0.2,log=True),
            'num_leaves':trial.suggest_int('num_leaves',16,128)}
    clf=lgb.LGBMClassifier(**params)
    pipe=Pipeline([('scaler',StandardScaler()),('clf',clf)])
    tscv=TimeSeriesSplit(n_splits=5)
    return cross_val_score(pipe,X,y,cv=tscv,scoring='accuracy').mean()

def run(ticker,horizon,trials,out):
    df=yf.download(ticker+'.NS',period='4y')
    X,y,_=make_features(df,horizon=horizon)
    study=optuna.create_study(direction='maximize')
    study.optimize(lambda tr: objective(tr,X,y),n_trials=trials)
    print('Best:',study.best_params)
    clf=lgb.LGBMClassifier(**study.best_params)
    pipe=Pipeline([('scaler',StandardScaler()),('clf',clf)])
    pipe.fit(X,y)
    os.makedirs(os.path.dirname(out),exist_ok=True)
    joblib.dump(pipe,out)
    print('Saved to',out)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--ticker',required=True)
    ap.add_argument('--horizon',type=int,default=3)
    ap.add_argument('--trials',type=int,default=20)
    ap.add_argument('--out',default='models/model_lgb.pkl')
    a=ap.parse_args()
    run(a.ticker,a.horizon,a.trials,a.out)

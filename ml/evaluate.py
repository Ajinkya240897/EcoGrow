import argparse,joblib,yfinance as yf
from sklearn.metrics import accuracy_score
from ml.feature_engineering import make_features

def evaluate(ticker,model_path,horizon):
    df=yf.download(ticker+'.NS',period='4y')
    X,y,_=make_features(df,horizon=horizon)
    model=joblib.load(model_path)
    preds=model.predict(X)
    print('Accuracy:',accuracy_score(y,preds))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--ticker',required=True)
    ap.add_argument('--model',required=True)
    ap.add_argument('--horizon',type=int,default=3)
    a=ap.parse_args()
    evaluate(a.ticker,a.model,a.horizon)

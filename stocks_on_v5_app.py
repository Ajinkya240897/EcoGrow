import streamlit as st,os,joblib
from datetime import datetime
from robust_io import fetch_current_price,normalize_ticker_for_yf
from fmp_client import safe_fetch

st.set_page_config(page_title='EcoGrow',layout='centered')
st.markdown("<h1 style='text-align:center'>EcoGrow</h1>",unsafe_allow_html=True)

st.sidebar.header('Inputs')
ticker=st.sidebar.text_input('Ticker (no suffix)','RELIANCE')
fmp_key=st.sidebar.text_input('FMP API Key (optional)',type='password')
horizon=st.sidebar.selectbox('Prediction horizon',['3 days','15 days','1 month','3 months','6 months','1 year'])
if st.sidebar.button('Run Prediction'):
    price=fetch_current_price(ticker).get('price')
    st.subheader(f'{ticker} Prediction - {horizon}')
    st.markdown(f'**Current price:** {price if price else "N/A"}')
    # fundamentals
    if fmp_key:
        fmp=safe_fetch(ticker,fmp_key);prof=fmp.get('profile');met=fmp.get('metrics')
        if prof: st.markdown(f"**Company:** {prof.get('companyName')}"); st.markdown(prof.get('description',''))
        if met: st.markdown(f"**Fundamentals score:** 60/100 (example)") 
    st.markdown('**Recommendation:** Buy/Hold/Sell with descriptive steps here.')
else:
    st.info('Enter inputs and click Run Prediction')

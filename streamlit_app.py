import streamlit
import yfinance
import datetime
import sklearn
import tensorflow
import numpy
import time
import GoogleNews
from nltk.sentiment.vader import SentimentIntensityAnalyzer

if 'tracked_stocks' not in streamlit.session_state:
    streamlit.session_state.tracked_stocks = []

streamlit.set_page_config(page_title = 'CrocStocks Stock Predictor', layout = 'wide')
streamlit.sidebar.header('Add Stocks to Watchlist')
ticker = streamlit.sidebar.text_input('Enter Stock Ticker(default AAPL)', value = 'AAPL')
start_date = streamlit.sidebar.date_input('Start Date(default one year)', value = datetime.datetime.now() - datetime.timedelta(days = 365))
end_date = streamlit.sidebar.date_input('End Date(default one year)', value = datetime.datetime.now())

if streamlit.sidebar.button('Add to Watchlist'): 
    if len(ticker.upper().strip()) == 0: 
        streamlit.sidebar.error('Please enter a valid stock ticker.')
    elif ticker.upper() in streamlit.session_state.tracked_stocks: 
        streamlit.sidebar.warning(f"{ticker.upper()} is already in your watchlist.")
    elif ticker.upper() not in streamlit.session_state.tracked_stocks: 
        streamlit.session_state.tracked_stocks.append(ticker.upper())
        streamlit.sidebar.success(f'You have added {ticker.upper()} to your watchlist!')

streamlit.sidebar.subheader('Your Watchlist')
for stock in streamlit.session_state.tracked_stocks: 
    streamlit.sidebar.markdown("---")
    close_diff = yfinance.download(stock, start = datetime.datetime.now() - datetime.timedelta(days = 1), end = datetime.datetime.now(), interval = '1d', progress = False, auto_adjust = True)
    if not close_diff.empty: 
        diff = (float(close_diff['Open'].iloc[0]) - float(close_diff['Close'].iloc[-1])) / float(close_diff['Open'].iloc[0])
        color = 'green' if diff >= 0 else 'red'
        change = diff * 100
        streamlit.sidebar.markdown(f"{stock} <span style = 'color: {color}; font-weight:bold;'>({change:.3f}%)</span>", unsafe_allow_html = True)
    else: 
        streamlit.sidebar.markdown(f"{stock} <span style = 'color: gray;'>(no data)</span>", unsafe_allow_html = True)
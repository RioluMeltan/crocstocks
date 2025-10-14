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
ticker = streamlit.sidebar.text_input('Enter Stock Ticker', value = 'AAPL')
start_date = streamlit.sidebar.date_input('Start Date', value = datetime.datetime.now() - datetime.timedelta(days = 365))
end_date = streamlit.sidebar.date_input('End Date', value = datetime.datetime.now())
if streamlit.sidebar.button('Add to Watchlist'):
    if ticker.upper() not in streamlit.session_state.tracked_stocks:
        streamlit.session_state.tracked_stocks.append(ticker.upper())
        streamlit.sidebar.success(f'You have added {ticker.upper()} to your watchlist!')
    elif ticker.upper() in streamlit.session_state.tracked_stocks:
        streamlit.sidebar.warning(f"{ticker.upper()} is already in your watchlist.")

streamlit.sidebar.subheader('Your Watchlist')
for stock in streamlit.session_state.tracked_stocks:
    streamlit.sidebar.write(f'{stock}')
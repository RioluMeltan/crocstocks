import streamlit
import yfinance
import datetime
import sklearn
import tensorflow
import numpy
import time
import GoogleNews
import plotly
from nltk.sentiment.vader import SentimentIntensityAnalyzer

if 'tracked_stocks' not in streamlit.session_state:
    streamlit.session_state.tracked_stocks = []

streamlit.set_page_config(page_title = 'CrocStocks Stock Predictor', layout = 'wide')
streamlit.sidebar.header('Add Stocks to Watchlist')
ticker = streamlit.sidebar.text_input('Enter Stock Ticker(default AAPL)', value = 'AAPL')

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
    close_diff = yfinance.download(stock, start = datetime.datetime.now() - datetime.timedelta(days = 1), end = datetime.datetime.now(), interval = '1d', progress = False, auto_adjust = True)
    if not close_diff.empty: 
        diff = (float(close_diff['Open'].iloc[0]) - float(close_diff['Close'].iloc[-1])) / float(close_diff['Open'].iloc[0])
        color = 'green' if diff >= 0 else 'red'
        change = diff * 100
        streamlit.sidebar.markdown(f"<div style = 'border:1px solid #ccc; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><strong>{stock}</strong><br><span style = 'color: {color}; font-weight: bold;'>Change: {change:.2f}%</span></div>", unsafe_allow_html = True)
    else: 
        streamlit.sidebar.markdown(f"{stock} <span style = '<div style = 'border:1px solid #ccc; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><strong>{stock}</strong><br><span style = 'color: gray; font-weight: bold;'>Change: N/A</span></div>", unsafe_allow_html = True)

streamlit.title('Your Watchlist Performance')
for stock in streamlit.session_state.tracked_stocks: 
    streamlit.subheader(f"{stock} - Historical Data")
    data = yfinance.download(stock, start = datetime.datetime.now() - datetime.timedelta(days = 365), end = datetime.datetime.now(), interval = '1d', progress = False, auto_adjust = True)
    if not data.empty:
        streamlit.line_chart(data['Close'])
    else:
        streamlit.warning(f"No data available for {stock}")
    if streamlit.button('5D'): 
        if not data.empty:
            streamlit.line_chart(data['Close'][359:364])
        else:
            streamlit.warning(f"No data available for {stock}")
    if streamlit.button('1M'): 
        if not data.empty:
            streamlit.line_chart(data['Close'][334:364])
        else:
            streamlit.warning(f"No data available for {stock}")
    if streamlit.button('6M'): 
        if not data.empty:
            streamlit.line_chart(data['Close'][184:364])
        else:
            streamlit.warning(f"No data available for {stock}")
    if streamlit.button('1Y'): 
        if not data.empty:
            streamlit.line_chart(data['Close'])
        else:
            streamlit.warning(f"No data available for {stock}")
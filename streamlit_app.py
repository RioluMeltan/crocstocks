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

ticker = streamlit.sidebar.text_input('Enter Stock Ticker')
if streamlit.sidebar.button('Add to Watchlist'): 
    if len(ticker.upper().strip()) == 0: 
        streamlit.sidebar.error('Please enter a valid stock ticker.')
    elif ticker.upper() in streamlit.session_state.tracked_stocks: 
        streamlit.sidebar.warning(f"{ticker.upper()} is already in your watchlist.")
    elif ticker.upper() not in streamlit.session_state.tracked_stocks: 
        streamlit.session_state.tracked_stocks.append(ticker.upper())
        streamlit.sidebar.success(f'You have added {ticker.upper()} to your watchlist!')

streamlit.sidebar.subheader('Your Watchlist')
if len(streamlit.session_state.tracked_stocks) == 0: 
    streamlit.sidebar.caption('Your watchlist is empty.')

@streamlit.cache_data
def get_change_data(ticker): 
    return yfinance.download(ticker, start = datetime.datetime.now() - datetime.timedelta(days = 1), end = datetime.datetime.now(), interval = '1d', progress = False, auto_adjust = True)

for stock in streamlit.session_state.tracked_stocks: 
    close_diff = get_change_data(stock)
    if not close_diff.empty: 
        diff = (float(close_diff['Open'].iloc[0]) - float(close_diff['Close'].iloc[-1])) / float(close_diff['Open'].iloc[0])
        color = 'green' if diff >= 0 else 'red'
        change = diff * 100
        streamlit.sidebar.markdown(f"<div style = 'border: 1px solid #ccc; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><strong>{stock}</strong><br><span style = 'color: {color};'>{'+' if diff >= 0 else ''}{(float(close_diff['Open'].iloc[0]) - float(close_diff['Close'].iloc[-1])):.2f}({'+' if diff >= 0 else ''}{change:.2f}%)</span></div>", unsafe_allow_html = True)
    else: 
        streamlit.sidebar.markdown(f"<div style = 'border: 1px solid #ccc; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><strong>{stock}</strong><br><span style = 'color: gray;'>Unavailable change amount</span></div>", unsafe_allow_html = True)

streamlit.title("Your Watchlist's Performance")
if len(streamlit.session_state.tracked_stocks) == 0: 
    streamlit.caption('Your watchlist is empty.')

@streamlit.cache_data
def get_long_data(ticker, days): 
    ipo_date = yfinance.Ticker(ticker).history(period = 'max').index[0].to_pydatetime()
    if ipo_date.replace(tzinfo = None) < datetime.datetime.now() - datetime.timedelta(days = days): 
        return yfinance.download(ticker, start = datetime.datetime.now() - datetime.timedelta(days = days), end = datetime.datetime.now(), interval = '1d', progress = False, auto_adjust = True)
    else: 
        return yfinance.download(ticker, start = ipo_date, end = datetime.datetime.now(), interval = '1d', progress = False, auto_adjust = True)
    
for stock in streamlit.session_state.tracked_stocks: 
    streamlit.subheader(f'{stock} - Historical Data')
    selections = {'Previous 5 Days': 5, 'Previous Month': 30, 'Previous 6 Months': 180, 'Previous Year': 365, 'All Time': 9999}
    selected_range = streamlit.radio('Select time range: ', list(selections.keys()), index = 4, horizontal = True, key = f'range_{stock}')
    data = get_long_data(stock, selections[selected_range])
    streamlit.line_chart(data['Close'])
import streamlit
import yfinance
import datetime
import sklearn
import tensorflow
import numpy
import time
import GoogleNews
from nltk.sentiment.vader import SentimentIntensityAnalyzer

streamlit.set_page_config(page_title = 'CrocStocks Stock Predictor', layout = 'wide')
streamlit.sidebar.header('Selected Stocks')
ticker = streamlit.sidebar.text_input('Enter Stock Ticker', value = 'AAPL')
start_date = streamlit.sidebar.date_input('Start Date', value = datetime.datetime.now() - datetime.timedelta(days = 365))
end_date = streamlit.sidebar.date_input('End Date', value = datetime.datetime.now())
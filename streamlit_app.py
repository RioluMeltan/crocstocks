import streamlit
import yfinance
import datetime
import sklearn
import tensorflow
import numpy
import pandas
import time
import GoogleNews
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class ProgressCallback(tensorflow.keras.callbacks.Callback): 
    def __init__(self, prog_bar, total_epochs, total, curr):
        super().__init__()
        self.prog_bar = prog_bar
        self.total_epochs = total_epochs
        self.total = total
        self.curr = curr
        self.last_epoch = 0

    def on_epoch_end(self, epoch, logs = None):
        self.last_epoch = epoch + 1
        self.prog_bar.progress(self.curr + int(((self.last_epoch / self.total_epochs) * 100) / self.total))

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
    if days == 99999: 
        if ipo_date.replace(tzinfo = None) < datetime.datetime.now() - datetime.timedelta(days = days): 
            return yfinance.download(ticker, start = datetime.datetime.now() - datetime.timedelta(days = days), end = datetime.datetime.now(), interval = '1mo', progress = False, auto_adjust = True)
        else: 
            return yfinance.download(ticker, start = ipo_date, end = datetime.datetime.now(), interval = '1mo', progress = False, auto_adjust = True)
    elif days == 1825: 
        if ipo_date.replace(tzinfo = None) < datetime.datetime.now() - datetime.timedelta(days = days): 
            return yfinance.download(ticker, start = datetime.datetime.now() - datetime.timedelta(days = days), end = datetime.datetime.now(), interval = '1wk', progress = False, auto_adjust = True)
        else: 
            return yfinance.download(ticker, start = ipo_date, end = datetime.datetime.now(), interval = '1wk', progress = False, auto_adjust = True)
    else: 
        if ipo_date.replace(tzinfo = None) < datetime.datetime.now() - datetime.timedelta(days = days): 
            return yfinance.download(ticker, start = datetime.datetime.now() - datetime.timedelta(days = days), end = datetime.datetime.now(), interval = '1d', progress = False, auto_adjust = True)
        else: 
            return yfinance.download(ticker, start = ipo_date, end = datetime.datetime.now(), interval = '1d', progress = False, auto_adjust = True)

def rmse(y_label, y_pred):
    return tensorflow.sqrt(tensorflow.reduce_mean(tensorflow.square(y_pred - y_label)))

def analyze_fundamentals(symbol, prog_bar, total): 
    start = time.perf_counter()
    fundamental_value = 0
    stock_info = yfinance.Ticker(symbol).info
    curr = 0
    try: 
        fundamental_value += stock_info.get('trailingEps', None) * 0.2
    except Exception as ex: 
        print(ex)
    prog_bar.progress(int((100 / total) / 7))
    curr += int((100 / total) / 7)
    try: 
        fundamental_value += stock_info.get('revenueGrowth', None) * 0.2
    except Exception as ex: 
        print(ex)
    prog_bar.progress(curr + int((100 / total) / 7))
    curr += int((100 / total) / 7)
    try: 
        fundamental_value += stock_info.get('profitMargins', None) * 0.15
    except Exception as ex: 
        print(ex)
    prog_bar.progress(curr + int((100 / total) / 7))
    curr += int((100 / total) / 7)
    try: 
        fundamental_value += stock_info.get('returnOnEquity', None) * 0.15
    except Exception as ex: 
        print(ex)
    prog_bar.progress(curr + int((100 / total) / 7))
    curr += int((100 / total) / 7)
    try: 
        fundamental_value -= stock_info.get('trailingPE', None) * 0.1
    except Exception as ex: 
        print(ex)
    prog_bar.progress(curr + int((100 / total) / 7))
    curr += int((100 / total) / 7)
    try: 
        fundamental_value += stock_info.get('dividendYield', None) * 0.1
    except Exception as ex: 
        print(ex)
    prog_bar.progress(curr + int((100 / total) / 7))
    curr += int((100 / total) / 7)
    try: 
        fundamental_value -= stock_info.get('debtToEquity', None) * 0.1
    except Exception as ex: 
        print(ex)
    prog_bar.progress(curr + int((100 / total) / 7))
    print(f"Fundamental analysis took {time.perf_counter() - start} seconds")
    return [fundamental_value, curr + int((100 / total) / 7)]

def fetch_sentiment(symbol): 
    start = time.perf_counter()
    news = GoogleNews.GoogleNews(lang = 'en', period = '7d')
    news.search(symbol)
    headlines = [item['title'] for item in news.result() if item.get('title')]
    paragraphs = [body['desc'] for body in news.result() if body.get('title')]
    sia = SentimentIntensityAnalyzer(lexicon_file = 'sentiment/vader_lexicon/vader_lexicon.txt')
    sentiment_scores = [sia.polarity_scores(headline)['compound'] for headline in headlines]
    sentiment_scores += [sia.polarity_scores(paragraph)['compound'] for paragraph in paragraphs]
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    print(f"Fetching market sentiment took {time.perf_counter() - start} seconds")
    return avg_sentiment

def historical_analysis(symbol, _prog_bar, total, curr): 
    data = yfinance.download(symbol, start = datetime.datetime.now() - datetime.timedelta(days = 365), end = datetime.datetime.now(), interval = '1d', progress = False, auto_adjust = True)
    start = time.perf_counter()
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range = (0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values)
    train_data = []
    for i in range(60, len(scaled_data)):
        train_data.append(scaled_data[i - 60:i, 0])
    train_data = numpy.array(train_data)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(train_data[:, :-1], train_data[:, -1], test_size = 0.2, random_state = 42)
    x_train = numpy.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = numpy.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    model = tensorflow.keras.models.Sequential([tensorflow.keras.Input(shape = (x_train.shape[1], 1)), tensorflow.keras.layers.LSTM(units = 50, return_sequences = True), tensorflow.keras.layers.LSTM(units = 50), tensorflow.keras.layers.Dense(units = 1)])
    model.compile(optimizer = tensorflow.keras.optimizers.Adam(learning_rate = 0.001), loss = rmse)
    early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
    model.fit(x_train, y_train, epochs = 50, batch_size = 32, validation_data = (x_test, y_test), callbacks = [early_stopping, ProgressCallback(_prog_bar, 50, total, curr)])
    predicted_stock_price = model.predict(scaled_data[-59:].reshape(1, -1, 1))
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    print(f"Historical analysis took {time.perf_counter() - start} seconds")
    return float(predicted_stock_price[0][0])

@streamlit.dialog('Quick Analysis Results', width = 'medium', dismissible = False)
def quick(f_true, s_true, h_true, stock): 
    progress_struct = streamlit.empty()
    progress = progress_struct.progress(0)
    if f_true: 
        f_results = analyze_fundamentals(stock, progress, sum([f_true, s_true, h_true]))
    if s_true: 
        s_results = fetch_sentiment(stock)
        progress.progress(f_results[1] + int(100 / sum([f_true, s_true, h_true])))
    if h_true:
        h_results = historical_analysis(stock, progress, sum([f_true, s_true, h_true]), f_results[1] + int(100 / sum([f_true, s_true, h_true])))
    progress.progress(100)
    progress_struct.empty()
    streamlit.code(f'{stock} Fundamentals: {f_results[0]:.4f}\n{stock} Market Sentiment: {s_results:.4f}\n{stock} Projected Next Day Close: US${h_results:.4f}')
    if streamlit.button('Close'): 
        streamlit.session_state.quick_open = False
        streamlit.rerun()

for stock in streamlit.session_state.tracked_stocks: 
    streamlit.subheader(f'{stock} - Historical Data')
    selections = {'Previous 5 Days': 5, 'Previous Month': 30, 'Previous 6 Months': 180, 'Previous Year': 365, 'Previous 5 Years': 1825, 'All Time': 99999}
    selected_range = streamlit.radio('Select time range: ', list(selections.keys()), index = 5, horizontal = True, key = f'range_{stock}')
    try: 
        col_1, col_2 = streamlit.columns([4, 1])
        with col_1: 
            data = get_long_data(stock, selections[selected_range])
            streamlit.line_chart(data['Close'])
        with col_2: 
            fundamental_check = streamlit.checkbox('Include Fundamentals', key = 'f' + stock)
            sentiment_check = streamlit.checkbox('Include Sentiment', key = 's' + stock)
            historical_check = streamlit.checkbox('Include Stock Prediction', key = 'h' + stock)
            if not fundamental_check and not sentiment_check and not historical_check:
                streamlit.markdown("<button disabled style = 'opacity:0.6;'>Quick Analysis</button>", unsafe_allow_html = True)
                streamlit.markdown("<button disabled style = 'opacity:0.6;'>Comprehensive Analysis</button>", unsafe_allow_html = True)
            else: 
                if streamlit.button('Quick Analysis'): 
                    streamlit.session_state.quick_open = True
                if streamlit.button('Comprehensive Analysis'): 
                    print('comprehensive')
    except Exception as exc: 
        streamlit.error('Something went wrong. Ensure your stock ticker is entered correctly and try reloading the page.')

if streamlit.session_state.quick_open: 
    quick(fundamental_check, sentiment_check, historical_check, stock)
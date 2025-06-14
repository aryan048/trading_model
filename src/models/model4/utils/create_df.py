from src.models.utils.sp_scraper import SPScraper
import yfinance as yf
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import talib as ta
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
import os
from yfinance.exceptions import YFTickerMissingError
from functools import lru_cache
from multiprocessing import Pool, cpu_count, get_context
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_ticker(ticker, ticker_data, train_data_till=None, test_data_from=None):
    ticker_data = ticker_data.dropna(inplace=False)

    # Ensuring is a copy of itself
    ticker_data = ticker_data.copy()

    # Process data
    ticker_data.reset_index(inplace=True)
    ticker_data.columns = ticker_data.columns.str.lower()
    
    # Make date a datetime, only year, month, day
    ticker_data['date'] = pd.to_datetime(ticker_data['date'], errors='coerce')

    
    if train_data_till:
        ticker_data = ticker_data[ticker_data['date'].dt.year <= int(train_data_till)]

    if test_data_from:
        ticker_data = ticker_data[ticker_data['date'].dt.year >= int(test_data_from)]

    if ticker_data.empty:
        return None, None

    ticker_data.loc[:,'ticker'] = ticker
    ticker_data.loc[:,'log_return_30d'] = np.log(ticker_data['close'].shift(-21) / ticker_data['close'])

    # Technicals no z score
    ticker_data.loc[:,'rsi'] = ta.RSI(ticker_data['close'], timeperiod=20)
    pattern_functions = {
                'CDL2CROWS': ta.CDL2CROWS,
                'CDL3BLACKCROWS': ta.CDL3BLACKCROWS,
                'CDL3INSIDE': ta.CDL3INSIDE,
                'CDL3LINESTRIKE': ta.CDL3LINESTRIKE,
                'CDL3OUTSIDE': ta.CDL3OUTSIDE,
                'CDL3STARSINSOUTH': ta.CDL3STARSINSOUTH,
                'CDL3WHITESOLDIERS': ta.CDL3WHITESOLDIERS,
                'CDLABANDONEDBABY': ta.CDLABANDONEDBABY,
                'CDLADVANCEBLOCK': ta.CDLADVANCEBLOCK,
                'CDLBELTHOLD': ta.CDLBELTHOLD,
                'CDLBREAKAWAY': ta.CDLBREAKAWAY,
                'CDLCLOSINGMARUBOZU': ta.CDLCLOSINGMARUBOZU,
                'CDLCONCEALBABYSWALL': ta.CDLCONCEALBABYSWALL,
                'CDLCOUNTERATTACK': ta.CDLCOUNTERATTACK,
                'CDLDARKCLOUDCOVER': ta.CDLDARKCLOUDCOVER,
                'CDLDOJI': ta.CDLDOJI,
                'CDLDOJISTAR': ta.CDLDOJISTAR,
                'CDLDRAGONFLYDOJI': ta.CDLDRAGONFLYDOJI,
                'CDLENGULFING': ta.CDLENGULFING,
                'CDLEVENINGDOJISTAR': ta.CDLEVENINGDOJISTAR,
                'CDLEVENINGSTAR': ta.CDLEVENINGSTAR,
                'CDLGAPSIDESIDEWHITE': ta.CDLGAPSIDESIDEWHITE,
                'CDLGRAVESTONEDOJI': ta.CDLGRAVESTONEDOJI,
                'CDLHAMMER': ta.CDLHAMMER,
                'CDLHANGINGMAN': ta.CDLHANGINGMAN,
                'CDLHARAMI': ta.CDLHARAMI,
                'CDLHARAMICROSS': ta.CDLHARAMICROSS,
                'CDLHIGHWAVE': ta.CDLHIGHWAVE,
                'CDLHIKKAKE': ta.CDLHIKKAKE,
                'CDLHIKKAKEMOD': ta.CDLHIKKAKEMOD,
                'CDLHOMINGPIGEON': ta.CDLHOMINGPIGEON,
                'CDLIDENTICAL3CROWS': ta.CDLIDENTICAL3CROWS,
                'CDLINNECK': ta.CDLINNECK,
                'CDLINVERTEDHAMMER': ta.CDLINVERTEDHAMMER,
                'CDLKICKING': ta.CDLKICKING,
                'CDLKICKINGBYLENGTH': ta.CDLKICKINGBYLENGTH,
                'CDLLADDERBOTTOM': ta.CDLLADDERBOTTOM,
                'CDLLONGLEGGEDDOJI': ta.CDLLONGLEGGEDDOJI,
                'CDLLONGLINE': ta.CDLLONGLINE,
                'CDLMARUBOZU': ta.CDLMARUBOZU,
                'CDLMATCHINGLOW': ta.CDLMATCHINGLOW,
                'CDLMATHOLD': ta.CDLMATHOLD,
                'CDLMORNINGDOJISTAR': ta.CDLMORNINGDOJISTAR,
                'CDLMORNINGSTAR': ta.CDLMORNINGSTAR,
                'CDLONNECK': ta.CDLONNECK,
                'CDLPIERCING': ta.CDLPIERCING,
                'CDLRICKSHAWMAN': ta.CDLRICKSHAWMAN,
                'CDLRISEFALL3METHODS': ta.CDLRISEFALL3METHODS,
                'CDLSEPARATINGLINES': ta.CDLSEPARATINGLINES,
                'CDLSHOOTINGSTAR': ta.CDLSHOOTINGSTAR,
                'CDLSHORTLINE': ta.CDLSHORTLINE,
                'CDLSPINNINGTOP': ta.CDLSPINNINGTOP,
                'CDLSTALLEDPATTERN': ta.CDLSTALLEDPATTERN,
                'CDLSTICKSANDWICH': ta.CDLSTICKSANDWICH,
                'CDLTAKURI': ta.CDLTAKURI,
                'CDLTASUKIGAP': ta.CDLTASUKIGAP,
                'CDLTHRUSTING': ta.CDLTHRUSTING,
                'CDLTRISTAR': ta.CDLTRISTAR,
                'CDLUNIQUE3RIVER': ta.CDLUNIQUE3RIVER,
                'CDLUPSIDEGAP2CROWS': ta.CDLUPSIDEGAP2CROWS,
                'CDLXSIDEGAP3METHODS': ta.CDLXSIDEGAP3METHODS
            }

    for pattern_name, pattern_function in pattern_functions.items():
        ticker_data[pattern_name] = pattern_function(
            ticker_data['open'],
            ticker_data['high'],
            ticker_data['low'],
            ticker_data['close']
        )

    # Technical z score
    ticker_data['sma_10'] = ta.SMA(ticker_data['close'], timeperiod=10)
    ticker_data['sma_30'] = ta.SMA(ticker_data['close'], timeperiod=30)
    ticker_data['macd'] = ta.MACD(ticker_data['close'], fastperiod=12, slowperiod=26, signalperiod=9)[0]
    ticker_data['macd_signal'] = ta.MACD(ticker_data['close'], fastperiod=12, slowperiod=26, signalperiod=9)[1]
    ticker_data['macd_hist'] = ta.MACD(ticker_data['close'], fastperiod=12, slowperiod=26, signalperiod=9)[2]
    
    # Z score vals
    # Calculate z-score for the last 60 days
    ticker_data['z_score_close'] = np.nan
    ticker_data['z_score_sma10'] = np.nan
    ticker_data['z_score_sma30'] = np.nan
    ticker_data['z_score_macd'] = np.nan
    ticker_data['z_score_macd_signal'] = np.nan
    ticker_data['z_score_macd_hist'] = np.nan
    for i in range(60, len(ticker_data)):
        # Z-score for close price
        window = ticker_data['close'].iloc[i-60:i]
        mean = window.mean()
        std = window.std()
        if std != 0:  # Avoid division by zero
            ticker_data.loc[ticker_data.index[i], 'z_score_close'] = (ticker_data['close'].iloc[i] - mean) / std
        
        # Z-score for SMA10
        window_sma10 = ticker_data['sma_10'].iloc[i-60:i]
        mean_sma10 = window_sma10.mean()
        std_sma10 = window_sma10.std()
        if std_sma10 != 0:
            ticker_data.loc[ticker_data.index[i], 'z_score_sma10'] = (ticker_data['sma_10'].iloc[i] - mean_sma10) / std_sma10
        
        # Z-score for SMA30
        window_sma30 = ticker_data['sma_30'].iloc[i-60:i]
        mean_sma30 = window_sma30.mean()
        std_sma30 = window_sma30.std()
        if std_sma30 != 0:
            ticker_data.loc[ticker_data.index[i], 'z_score_sma30'] = (ticker_data['sma_30'].iloc[i] - mean_sma30) / std_sma30

        # Z-score for MACD
        window_macd = ticker_data['macd'].iloc[i-60:i]
        mean_macd = window_macd.mean()
        std_macd = window_macd.std()
        if std_macd != 0:
            ticker_data.loc[ticker_data.index[i], 'z_score_macd'] = (ticker_data['macd'].iloc[i] - mean_macd) / std_macd

        # Z-score for MACD Signal
        window_macd_signal = ticker_data['macd_signal'].iloc[i-60:i]
        mean_macd_signal = window_macd_signal.mean()
        std_macd_signal = window_macd_signal.std()
        if std_macd_signal != 0:
            ticker_data.loc[ticker_data.index[i], 'z_score_macd_signal'] = (ticker_data['macd_signal'].iloc[i] - mean_macd_signal) / std_macd_signal

        # Z-score for MACD Histogram
        window_macd_hist = ticker_data['macd_hist'].iloc[i-60:i]
        mean_macd_hist = window_macd_hist.mean()
        std_macd_hist = window_macd_hist.std()
        if std_macd_hist != 0:
            ticker_data.loc[ticker_data.index[i], 'z_score_macd_hist'] = (ticker_data['macd_hist'].iloc[i] - mean_macd_hist) / std_macd_hist

    feature_cols = ['log_return_30d', 'rsi', 'z_score_close', 'z_score_sma10', 
                    'z_score_sma30', 'z_score_macd', 'z_score_macd_signal', 'z_score_macd_hist'] + list(pattern_functions.keys())
    
    return ticker_data, feature_cols

# Define a wrapper function for multiprocessing
def process_ticker_wrapper(args):
    return process_ticker(*args)

def create_df(data_len="max", train_data_till=None, test_data_from=None):
    # Replace '.' with '-' in ticker symbols, also add SPY as a benchmark
    scraper = SPScraper()

    sp_tickers = [ticker.replace(".", "-") for ticker in sorted(scraper.scrape_sp500_symbols().index)] + ["^GSPC"]
    data_frames = []

    # Download data
    data = yf.download(sp_tickers, period=data_len)

    # Split the DataFrame by ticker with progress tracking
    ticker_dfs = {}
    tickers = data.columns.levels[1]
    for ticker in tqdm(tickers, desc="Splitting data by ticker", unit="ticker"):
        ticker_dfs[ticker] = data.xs(ticker, axis=1, level=1)

    # Use multiprocessing.Pool with "fork" context
    args = [(ticker, ticker_df, train_data_till, test_data_from) for ticker, ticker_df in ticker_dfs.items()]
    with get_context("fork").Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(process_ticker_wrapper, args), total=len(args), desc="Adding features", unit="ticker", leave=True))
        for result in results:
            try:
                ticker_data, feature_cols = result
                if ticker_data is not None:
                    data_frames.append(ticker_data)
            except Exception as e:
                print(f"Error processing ticker: {e}")

    # Combine all dataframes
    data = pd.concat(data_frames, ignore_index=True)
    del data_frames
    # Get ^VIX and join it to data based on date
    vix = yf.Ticker("^VIX").history(period=data_len)
    # Make "Date" a column
    vix.reset_index(inplace=True)
    # Make all column names lowercase
    vix.columns = vix.columns.str.lower()
    # Rename the "close" column to "vix"
    vix.rename(columns={'close': 'vix'}, inplace=True)
    # Refine df to only include date and vix
    vix = vix[['date', 'vix']]
    # Make date a datetime, only year, month, day
    data['date'] = pd.to_datetime(data['date']).dt.date  # Ensure 'date' in data is of type datetime.date
    vix['date'] = pd.to_datetime(vix['date']).dt.date

    # Merge VIX data with main dataframe based on date
    data = pd.merge(data, vix, on='date', how='left')
    pd.set_option('display.max_columns', None)

    # Prepare the data for the model
    if train_data_till:
        df_scaler = StandardScaler()
        label_encoder = LabelEncoder()
        
        # Fit the label encoder on the sp_tickers list
        sp_tickers = [ticker.replace(".", "-") for ticker in sorted(scraper.scrape_sp500_symbols().index)] + ["^GSPC"]
        
        # Encode the tickers in the data based on the fitted label encoder
        data["encoded_ticker"] = data["ticker"].map(lambda x: label_encoder.fit_transform([x])[0] if x in sp_tickers else -1)

    else:
        try:
            df_scaler = joblib.load(f'/Users/aryanhazra/Downloads/Github Repos/trading_model/src/models/model4/{int(test_data_from) - 1}/scaler_df.pkl')
            label_encoder = joblib.load(f'/Users/aryanhazra/Downloads/Github Repos/trading_model/src/models/model4/{int(test_data_from) - 1}/label_encoder.pkl')
        except:
            fallback_scalers = (f'Scaler or label encoder for year {int(test_data_from) - 1} not found. Please select a training year scaler to use.')
            df_scaler = joblib.load(f'/Users/aryanhazra/Downloads/Github Repos/trading_model/src/models/model4/{fallback_scalers}/scaler_df.pkl')
            label_encoder = joblib.load(f'/Users/aryanhazra/Downloads/Github Repos/trading_model/src/models/model4/{fallback_scalers}/label_encoder.pkl')
        # Transform the tickers in the data based on the loaded label encoder
        data["encoded_ticker"] = data["ticker"].map(lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1)

    # Select and scale
    # Can't forget to scale the encoded ticker
    feature_cols.append("encoded_ticker")
    df_vals = data.filter(feature_cols)
    df_vals = df_vals.values

    if train_data_till:
        scaled_df_vals = df_scaler.fit_transform(df_vals)
    else:
        scaled_df_vals = df_scaler.transform(df_vals)

    for i, feature in enumerate(feature_cols):
        data[f'scaled_{feature}'] = scaled_df_vals[:, i]
        feature_cols[i] = f'scaled_{feature}'

    feature_cols.remove("scaled_log_return_30d")

    train_df = data.dropna(inplace=False)
    pred_df = data

    # Group the data by ticker
    train_grouped_dfs = train_df.groupby('ticker')
    train_grouped_dfs = {ticker: df.sort_values(by='date').reset_index(drop=True) for ticker, df in train_grouped_dfs}

    pred_grouped_dfs = pred_df.groupby('ticker')
    pred_grouped_dfs = {ticker: df.sort_values(by='date').reset_index(drop=True) for ticker, df in pred_grouped_dfs}

    return train_grouped_dfs, pred_grouped_dfs, df_scaler, label_encoder, feature_cols




        
if __name__ == "__main__":
    create_df()
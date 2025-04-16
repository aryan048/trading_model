from models.utils.sp_scraper import SPScraper
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


def process_ticker(ticker, data_len="max", train_data_till = None, test_data_from = None):
        try:
            # Retry logic
            while True:
                try:
                    ticker_data = yf.Ticker(ticker).history(period=data_len)
                    break
                except YFTickerMissingError as e:
                    print("Trying again...")
                    time.sleep(10)
                except Exception as e:
                    print("Trying again...")
                    time.sleep(10)

            if ticker_data.empty:
                return None, None

            # Process data
            ticker_data.reset_index(inplace=True)
            ticker_data.columns = ticker_data.columns.str.lower()
            ticker_data['date'] = pd.to_datetime(ticker_data['date'])

            if train_data_till:
                # TEsting smth delete later, only training on data before 2015, then test from 2016 onwards
                ticker_data = ticker_data[ticker_data['date'].dt.year <= int(train_data_till)]

            if test_data_from:
                ticker_data = ticker_data[ticker_data['date'].dt.year >= int(test_data_from)]

            if ticker_data.empty:
                return None, None
            

            ticker_data['ticker'] = ticker
            ticker_data['log_return_30d'] = np.log(ticker_data['close'].shift(-21) / ticker_data['close'])

            # Technicals no z score

            # Technical z score
            ticker_data['rsi'] = ta.RSI(ticker_data['close'], timeperiod=20)
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
                            'z_score_sma30', 'z_score_macd', 'z_score_macd_signal', 'z_score_macd_hist']

            # Make date a datetime, only year, month, day
            ticker_data['date'] = pd.to_datetime(ticker_data['date']).dt.date
            
            
            
            return ticker_data, feature_cols
        
        except Exception as e:
            print(f"Failed to process {ticker}: {e}")
            return None, None


def create_df(data_len="max", train_data_till = None, test_data_from = None):
    # Replace '.' with '-' in ticker symbols, also add SPY as a benchmark
    scraper = SPScraper()

    sp_tickers = [ticker.replace(".", "-") for ticker in sorted(scraper.scrape_sp500_symbols().index)] + ["^GSPC"]
    data_frames = []

    cpu_count = os.cpu_count() or 4  # Fallback if CPU count cannot be determined
    max_workers = min(cpu_count * 4, 60)  # Cap at 60 to avoid rate limiting


    # Use multithreading for I/O-bound operations like data download
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_ticker, ticker, data_len, train_data_till, test_data_from): ticker for ticker in sp_tickers}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading data", unit="ticker"):
            ticker_data, feature_cols = future.result()
            if ticker_data is not None:
                data_frames.append(ticker_data)

    # Combine all dataframes
    data = pd.concat(data_frames, ignore_index=True)
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
    vix['date'] = pd.to_datetime(vix['date']).dt.date

    # Merge VIX data with main dataframe based on date
    data = pd.merge(data, vix, on='date', how='left')
    pd.set_option('display.max_columns', None)

    # Prepare the data for the model
    if train_data_till:
        df_scaler = StandardScaler()
        label_encoder = LabelEncoder()
        data["encoded_ticker"] = label_encoder.fit_transform(data["ticker"])

    else:
        df_scaler = joblib.load('/Users/aryanhazra/Downloads/VSCode Repos/trading_model/src/models/model4/scaler_df.pkl')
        label_encoder = joblib.load('/Users/aryanhazra/Downloads/VSCode Repos/trading_model/src/models/model4/label_encoder.pkl')
        data["encoded_ticker"] = label_encoder.transform(data["ticker"])

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
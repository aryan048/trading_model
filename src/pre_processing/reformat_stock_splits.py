from sqlalchemy import create_engine, inspect
import pandas as pd
import requests
import os
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src import SMS_notifier
import time
from pre_processing import technical_indicators

def stock_split_api(ticker, max_retries=5, retry_delay=60):
    response = []

    for abool in ['true', 'false']:
        while True:
            try:
                # Attempt to fetch the data
                url = f"https://api.polygon.io/v3/reference/splits?ticker={ticker}&reverse_split={abool}&order=desc&limit=1000&sort=execution_date&apiKey={os.getenv('polygon_api_key')}"
                result = requests.get(url).json()['results']
                response += result
                break  # If successful, break out of the retry loop
            except Exception as e:
                time.sleep(5)  # Wait before retrying


    return response

def edit_db(stock_splits, ticker, ticker_df, engine):
    ticker_df['date'] = pd.to_datetime(ticker_df['date'])  # Ensure date column is in datetime format

    for split in stock_splits:
        execution_date = pd.to_datetime(split['execution_date'])  # Example execution date
        

        adjustment_factor = split['split_to'] / split['split_from']

        for val in ['open', 'close', 'high', 'low']:
            #avoid implicit data type conversion in future pandas
            ticker_df[val] = ticker_df[val].astype(float)  # Convert to float before division
            ticker_df.loc[ticker_df['date'] < execution_date, val] /= adjustment_factor

        # Avoid implicit data type conversion in future pandas
        ticker_df['volume'] = ticker_df['volume'].astype(float)  # Convert to float before multiplication
        ticker_df.loc[ticker_df['date'] < execution_date, 'volume'] *= adjustment_factor

        #call technical indicators
    technical_indicators.create_technical_indicators(ticker, ticker_df, engine)

def reformat_stock_splits(ticker, ticker_df, engine):

    stock_splits = stock_split_api(ticker)

    #save time if no stock splits (wont load the df)
    if stock_splits:
        stock_splits.sort(key=lambda x: x['execution_date'], reverse=True)
        edit_db(stock_splits, ticker, ticker_df, engine)

if __name__ == "__main__":
    reformat_stock_splits(['AAPL'])  # Example tickers, replace with actual ones        


import sys
import os
import requests

# Add the absolute path to the project root directory
project_root = "/Users/aryanhazra/Repos/trading_model"
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.utils.sp_scraper import SPScraper
from tqdm import tqdm
from multiprocessing import get_context, cpu_count
import pandas as pd
from src.models.model5.TalibIndicators import TalibIndicators
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, BatchNormalization, LeakyReLU, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from copy import deepcopy
import tensorflow as tf
import gc
import json
import yfinance as yf



# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
tf.get_logger().setLevel('ERROR')         # Suppress TensorFlow INFO logs

scraper = SPScraper()
constituents_df, changes_df, historical_df = scraper.scrape_sp500_symbols()
all_tickers = list(set(historical_df['ticker_added'].dropna().tolist() + historical_df['ticker_removed'].dropna().tolist() + constituents_df['ticker_added'].dropna().tolist() + ['SPY']))

#sp_tickers = [ticker.replace(".", "-") for ticker in all_tickers] + ["SPY"]

win_count  = 0
lose_count = 0
wins = []
losses = []
returns = 1
spy_returns = 1
bias = 1



temp_data_dir = "src/models/model5/temp_data"
os.makedirs(temp_data_dir, exist_ok=True)

def download_ticker_data(ticker):
    """Download data for a single ticker"""
    try:
        ticker = ticker.replace(".", "-")
        url = f"https://eodhd.com/api/eod/{ticker}.US?period=d&api_token=680d28928681c6.96535696&fmt=json"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return {ticker: pd.DataFrame(data)}
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return {ticker: pd.DataFrame()}

def download_changes_ticker_data(args):
    """Download data for a single ticker from changes data"""
    ticker, blocked_tickers, delisted_tickers = args
    try:
        if ticker in blocked_tickers:
            return {ticker: pd.DataFrame()}

        if ticker in delisted_tickers.keys():
            old_ticker = delisted_tickers[ticker]['ticker']
            response = requests.get(f"https://eodhd.com/api/eod/{old_ticker}.US?period=d&api_token=680d28928681c6.96535696&fmt=json")
            if response.status_code == 200:
                data = response.json()
                return {ticker: pd.DataFrame(data)}
            else:
                return {ticker: pd.DataFrame()}
        else:
            response = requests.get(f"https://eodhd.com/api/eod/{ticker}.US?period=d&api_token=680d28928681c6.96535696&fmt=json")
            if response.status_code == 200:
                data = response.json()
                return {ticker: pd.DataFrame(data)}
            else:
                return {ticker: pd.DataFrame()}
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return {ticker: pd.DataFrame()}

def download_data() -> dict[str, pd.DataFrame]:
    dataframes = {}

    download_tickers = constituents_df['ticker_added'].tolist()
    download_tickers.append('SPY')
    # Use multiprocessing to download data in parallel
    with get_context('spawn').Pool(processes=cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(download_ticker_data, download_tickers),
            total=len(download_tickers),
            desc="Downloading constituent data"
        ))

    for result in results:
        for ticker, df in result.items():
            dataframes[ticker] = df
    
    blocked_tickers = ["DFS", "DISH", "SIVB", "NLSN", "DISCA", "PBCT", "KSU", "ETFC", "JWN", "TSS", "SRCL", "TWX", "BCR", "MNK", "SWN", "LM", "DO", "ESV", "GMCR", "JOY", "WIN", "BEAM", "WPX", "JCP", "NYX", "SAI", "DF", "FII", "RRD", "ANR", "KFT", "LXK", "WFR", "MDP", "JNY", "WB", "FRE", "CFC", "ABK", "QTRN", "LDW", "AN", "SUN", "USL", "HNG"]

    with open("selected_stocks_with_dates.json", "r") as f:
        delisted_tickers = json.load(f)

    # Collect all tickers from changes
    all_change_tickers = []
    for tickers in [changes_df['ticker_added'], changes_df['ticker_removed']]:
        all_change_tickers.extend(tickers.dropna().tolist())
    
    # Prepare arguments for multiprocessing
    args_list = [(ticker, blocked_tickers, delisted_tickers) for ticker in list(set(all_change_tickers))]
    
    # Use multiprocessing to download data in parallel
    with get_context('spawn').Pool(processes=cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(download_changes_ticker_data, args_list),
            total=len(args_list),
            desc="Downloading changes data"
        ))
    
    # Process results and handle delisted ticker merging
    for result in results:
        for ticker, df in result.items():
            if not df.empty:
                if ticker in dataframes.keys():
                    # Merge the old ticker data with the new ticker data
                    old_data = dataframes.get(ticker, pd.DataFrame())
                    new_data = df
                    
                    # Combine the dataframes, keeping all data from both
                    combined_data = pd.concat([old_data, new_data], ignore_index=True)
                    
                    # Remove duplicates based on date (assuming there's a date column)
                    if not combined_data.empty and 'date' in combined_data.columns:
                        combined_data = combined_data.drop_duplicates(subset=['date'], keep='first')
                        combined_data = combined_data.sort_values('date')
                    
                    dataframes[ticker] = combined_data
            else:
                dataframes[ticker] = df
        
    return dataframes
    
def calculate_z_score():
    pass

def add_features(ticker, df):
    # Make all columns lower case for easier indexing
    df.reset_index(inplace=True)
    df.columns = df.columns.str.lower()

    # Add ticker and return columns
    df['ticker'] = ticker
    df['return_21d'] = df['adjusted_close'].pct_change(periods=21, fill_method = None).shift(-21)

    # Add TA-Lib indicators
    Talib = TalibIndicators(df)
    df = Talib.df
    features = Talib.features

    return df, features

# Define a wrapper function for multiprocessing
def multithread_wrapper(args : tuple[str, pd.DataFrame]) -> pd.DataFrame:
    """Wrapper function to call add_features with unpacked arguments"""
    return add_features(*args)

def multithread_add_features(dataframes : dict[str, pd.DataFrame]):
    """Add features to the dataframes using multiprocessing"""
    full_df = pd.DataFrame()

    # Use multiprocessing.Pool with "fork" context
    args = [(ticker, ticker_df) for ticker, ticker_df in dataframes.items() if not ticker_df.empty]

    with get_context("fork").Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(multithread_wrapper, args), total=len(args), desc="Creating Multithread Processes", unit="ticker", leave=True))
        for result in tqdm(results, total=len(results), desc="Processing results", unit="Ticker", leave=True):
            try:
                df, features = result
                if df is not None:
                    full_df = pd.concat([full_df, df], ignore_index=True)                    
            except Exception as e:
                print(f"Error processing ticker: {e}")

    return full_df, features

def add_vix(full_df):
    # Get ^VIX and join it to data based on date
    vix = yf.Ticker("^VIX").history(period="max")
    # Make "Date" a column
    vix.reset_index(inplace=True)
    # Make all column names lowercase
    vix.columns = vix.columns.str.lower()
    # Rename the "close" column to "vix"
    vix.rename(columns={'close': 'vix'}, inplace=True)
    # Refine df to only include date and vix
    vix = vix[['date', 'vix']]
    # Make date a datetime, only year, month, day
    full_df['date'] = pd.to_datetime(full_df['date']).dt.date  # Ensure 'date' in data is of type datetime.date
    vix['date'] = pd.to_datetime(vix['date']).dt.date

    # Merge VIX data with main dataframe based on date
    data = pd.merge(full_df, vix, on='date', how='left')

    data.dropna(inplace=True)

    return data

def drop_data(dataframes, drop_date, pred_date = None, constituents_df = None):
    """Drops data after specified date (whatever index spy data is on in the loop) and before S&P addition"""
    pred_data = pd.DataFrame()
    dropped_dataframes = {}

    print(f"Expected: {len(constituents_df)}")
    skipped_tickers = []
    empty_tickers = []

    for ticker, df in tqdm(dataframes.items(), total=len(dataframes), desc="Dropping data", unit="ticker"):
        if ticker in constituents_df['ticker'].values:
            pred_idx = df.index[df['date'] == pred_date]
            if not pred_idx.empty:
                pred_data = pd.concat([pred_data, df.loc[df['date'] == pred_date]], ignore_index=True)
                dataframes[ticker].drop(df[df['date'] > drop_date].index, inplace=True)
                if not df.empty:
                    dropped_dataframes[ticker] = df
                else:
                    empty_tickers.append(ticker)                 
        else:
            skipped_tickers.append(ticker)

    print(f"After drop: {len(dropped_dataframes)}")
    return dropped_dataframes, pred_data


# def drop_data(dataframes, drop_date, pred_date = None):
#     """Drops data after specified date (whatever index spy data is on in the loop) and before S&P addition"""
#     pred_data = pd.DataFrame()
#     for ticker, df in tqdm(dataframes.items(), total=len(dataframes), desc="Dropping data", unit="ticker"):
#         # Get test data NOT Y TRAIN
#         dataframes[ticker].drop(df[df['date'] <= sp_addition_dates[ticker]].index, inplace=True)
#         if pred_date:
#             pred_idx = df.index[df['date'] == pred_date]
#             if not pred_idx.empty:
#                 pred_data = pd.concat([pred_data, df.loc[df['date'] == pred_date]], ignore_index=True)
#         dataframes[ticker].drop(df[df['date'] > drop_date].index, inplace=True)
#     dropped_dataframes = {ticker: df for ticker, df in dataframes.items() if not df.empty}

def z_score_data(dataframes, features):
    """Z-score the dataframes"""
    for ticker, df in tqdm(dataframes.items(), total=len(dataframes), desc="Z-scoring data", unit="ticker"):
        # Z-score the features
        for feature in features:
            rolling_mean = df[feature].rolling(window=60).mean()
            rolling_std = df[feature].rolling(window=60).std()

            # To avoid division by zero, set std to NaN where it's 0, then fill z-score with 0
            z_score = (df[feature] - rolling_mean) / rolling_std.replace(0, np.nan)
            dataframes[ticker][f"z_score_{feature}"] = z_score.fillna(0)

    z_score_features = [f"z_score_{feature}" for feature in features]
    z_score_dataframes = {ticker: df for ticker, df in dataframes.items()}
    return z_score_dataframes, z_score_features

def scale_data(dataframes, features, df_scaler=None, target_scaler=None):
    """Scale and label encode the dataframes"""
    label_encoder = LabelEncoder()

    df = pd.concat(dataframes.values(), ignore_index=True)

    df.dropna(inplace=True)
    
    label_encoder.fit(all_tickers)
    df["encoded_ticker"] = df["ticker"].map(lambda x: label_encoder.transform([x])[0] if x in all_tickers else -1)

    # Add non talib features
    features.append("encoded_ticker")
    features.append('vix')

    df_vals = df.filter(features).values

    target_vals = df['return_21d'].values.reshape(-1, 1)

    if not df_scaler and not target_scaler:
        df_scaler = StandardScaler()
        target_scaler = StandardScaler()
        scaled_df_vals = df_scaler.fit_transform(df_vals)
        scaled_target = target_scaler.fit_transform(target_vals)
    else:
        scaled_df_vals = df_scaler.transform(df_vals)
        scaled_target = target_scaler.transform(target_vals)

    # Optimized block to avoid DataFrame fragmentation
    global scaled_features
    scaled_features = [f'scaled_{feature}' for feature in features]
    scaled_df = pd.DataFrame(scaled_df_vals, columns=scaled_features)
    scaled_df = pd.concat([df.reset_index(drop=True), scaled_df.reset_index(drop=True)], axis=1)

    scaled_df['scaled_return_21d'] = scaled_target

    
    return scaled_df, scaled_features, df_scaler, target_scaler

def create_sliding_windows(dataframe, features):
    """Create sliding windows for the dataframes"""

    dataframes = dataframe.groupby('ticker')
    dataframes = {ticker: df.sort_values(by='date').reset_index(drop=True) for ticker, df in dataframes}

    validation_split = 0.1
    for ticker, df in tqdm(dataframes.items(), desc = "Creating sliding window", unit="ticker", total=len(dataframes)):
        local_x_train, local_y_train = [], []
        local_x_val, local_y_val = [], []

        split_index = int(len(df) * (1 - validation_split))

        # Split the data into training and validation sets
        train_df = df[:split_index]
        val_df = df[split_index:]

        # Process training data
        for i in range(60, len(train_df)):
            window = train_df.iloc[i - 60:i][features].values  # shape (60, num_features)
            local_x_train.append(window)
            local_y_train.append(train_df.iloc[i]['scaled_return_21d'])

        # Process validation data
        for i in range(60, len(val_df)):
            window = val_df.iloc[i - 60:i][features].values  # shape (60, num_features)
            local_x_val.append(window)
            local_y_val.append(val_df.iloc[i]['scaled_return_21d'])

        # Save training and validation data to disk
        if not any([local_x_train, local_y_train, local_x_val, local_y_val]):
            continue

        np.save(os.path.join(temp_data_dir, f"{ticker}_train_x.npy"), np.array(local_x_train))
        np.save(os.path.join(temp_data_dir, f"{ticker}_train_y.npy"), np.array(local_y_train))
        np.save(os.path.join(temp_data_dir, f"{ticker}_val_x.npy"), np.array(local_x_val))
        np.save(os.path.join(temp_data_dir, f"{ticker}_val_y.npy"), np.array(local_y_val))
    
def create_pred_sliding_window(scaled_dataframe, scaled_features):
    dataframes = scaled_dataframe.groupby('ticker')
    dataframes = {ticker: df.sort_values(by='date').reset_index(drop=True) for ticker, df in dataframes}

    x_pred = []
    for ticker, df in dataframes.items():
        if len(df) < 60:
            continue
        window = df.iloc[-60:][scaled_features].values

        x_pred.append([ticker,window])
    
    return x_pred

def train_test_loop(spy_df, dataframes, features):

    for i in range(67, len(spy_df) - 21, 1):
        # Clear any old sliding-window files so we only train on current window
        for f in os.listdir(temp_data_dir):
            os.remove(os.path.join(temp_data_dir, f))

        #Drop date is y train
        print(f"Training on date: {spy_df.iloc[i]['date']}")
        drop_date = spy_df.iloc[i]['date']
        pred_date = spy_df.iloc[i + 21]['date']
        spy_pred = spy_df.iloc[i + 21]['return_21d']

        #create curr consituents table on drop_date
        constituents_df = SPScraper().build_table(drop_date)


        # Passing a copy of the df so the original df is intact
        print(f"Before drop: {len(dataframes)}")
        dropped_dataframes, pred_df = drop_data(deepcopy(dataframes), drop_date, pred_date, constituents_df)

        z_score_dataframes, z_score_features = z_score_data(dropped_dataframes, features)

        scaled_dataframe, scaled_features, df_scaler, target_scaler = scale_data(z_score_dataframes, z_score_features)
        
        create_sliding_windows(scaled_dataframe, scaled_features)

        model = create_model(scaled_features)

        model = train_model(model)

        del scaled_dataframe, z_score_dataframes
        gc.collect()
        tf.keras.backend.clear_session()

        pred_drop_date = spy_df.iloc[i + 20]['date']

        # Dropping everything including the 21st day (the day we are predicting) so it doesnt get scaled
        dropped_dataframes, _ = drop_data(dataframes, pred_drop_date, constituents_df=constituents_df)

        z_score_dataframes, z_score_features = z_score_data(dropped_dataframes, features)

        scaled_dataframe, scaled_features, _, _ = scale_data(z_score_dataframes, z_score_features, df_scaler, target_scaler)

        x_pred = create_pred_sliding_window(scaled_dataframe, scaled_features)

        predictions = predict(x_pred, model, df_scaler, target_scaler, scaled_features)

        del model

        os.system('cls' if os.name == 'nt' else 'clear')    

        top_10_predictions = dict(sorted(predictions.items(), 
                                            key=lambda x: x[1], 
                                            reverse=True)[:10])

        all_snp_predictions = dict(sorted(predictions.items(), 
                                            key=lambda x: x[1], 
                                            reverse=True))
        local_bias = avg_real_return = np.mean([pred_df[pred_df['ticker'] == ticker]['return_21d'].values[0] for ticker in all_snp_predictions.keys()])

        
        avg_predicted_return = np.mean([pred for _, pred in top_10_predictions.items()])
        avg_real_return = np.mean([pred_df[pred_df['ticker'] == ticker]['return_21d'].values[0] for ticker in top_10_predictions.keys()])

        print(f"\nPredictions for date: {spy_df.iloc[i + 21]['date']}")
        print("Top 10 Predicted Returns:")
        print("Ticker | Predicted Return | Actual Return")
        print("-" * 45)
        for ticker, pred in top_10_predictions.items():
            # Convert to percentages by multiplying by 100
            print(f"{ticker:6} | {pred*100:13.2f}% | {pred_df[pred_df['ticker'] == ticker]['return_21d'].values[0]*100:.2f}%")
        print("\nAverage Predicted Return for Top 10: {:.2f}%".format(avg_predicted_return*100))
        print("\nAverage Actual Return for Top 10: {:.2f}%".format(avg_real_return*100))
        print(f"SPY Return: {spy_pred*100:.2f}%")

        global win_count, lose_count, wins, losses, returns, bias

        if avg_real_return > spy_pred:
            win_count += 1
            wins.append(avg_real_return - spy_pred)
        else:
            lose_count += 1
            losses.append(spy_pred - avg_real_return)

        global returns, spy_returns
        returns *= (1+avg_real_return)
        spy_returns *= (1+spy_pred)
        bias *= (1+local_bias)



        print(f"Win Count: {win_count}, Lose Count: {lose_count}")
        print(f"Average win by {np.mean(wins):.2f}")
        print(f"Average loss by {np.mean(losses):.2f}")
        print(f"Total return {((returns - 1) * 100):.2f}%")
        print(f"Spy returns {((spy_returns-1) * 100):.2f}%")
        print(f"Bias {((bias-1) * 100):.2f}%")


def create_model(scaled_features):
    model = Sequential()

    # Convolutional layers for local pattern extraction
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(60, len(scaled_features))))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # Stacked Bidirectional LSTM for capturing sequence relationships
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.3))

    # Dense layers for final nonlinear transformation
    model.add(Dense(128, kernel_regularizer=regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.4))  # Slightly increased dropout to reduce overfitting

    model.add(Dense(64, kernel_regularizer=regularizers.l2(0.001)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))

    model.add(Dense(1))  # Output log return prediction

    model.summary()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss=keras.losses.Huber(delta=1.0),  # Huber = better for stability on noisy targets
        metrics=[keras.metrics.RootMeanSquaredError()]
    )

    return model

def tf_data_generator(x_files, y_files, batch_size, shuffle_buffer=50_000):
    def generator():
        for x_file, y_file in zip(x_files, y_files):
            x_data = np.load(os.path.join(temp_data_dir, x_file))
            y_data = np.load(os.path.join(temp_data_dir, y_file))
            for i in range(len(x_data)):
                yield x_data[i], y_data[i]

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec((60, len(scaled_features)), tf.float32),
            tf.TensorSpec((), tf.float32)
        )
    )
    # shuffle only buffer_size at a time
    return ds.shuffle(shuffle_buffer)\
             .batch(batch_size)\
             .prefetch(tf.data.AUTOTUNE)

def data_generator_from_storage_split(temp_data_dir, batch_size):
    train_x_files = sorted([f for f in os.listdir(temp_data_dir) if f.endswith("_train_x.npy")])
    train_y_files = sorted([f for f in os.listdir(temp_data_dir) if f.endswith("_train_y.npy")])
    val_x_files = sorted([f for f in os.listdir(temp_data_dir) if f.endswith("_val_x.npy")])
    val_y_files = sorted([f for f in os.listdir(temp_data_dir) if f.endswith("_val_y.npy")])

    return train_x_files, train_y_files, val_x_files, val_y_files

def generator():
    train_x_files, train_y_files, val_x_files, val_y_files = data_generator_from_storage_split(temp_data_dir, batch_size=512)
    train_gen = tf_data_generator(train_x_files, train_y_files, batch_size=512)
    val_gen = tf_data_generator(val_x_files, val_y_files, batch_size=512)

    return train_gen, val_gen

def train_model(model):
    early_stopping = keras.callbacks.EarlyStopping(monitor='root_mean_squared_error', 
                                            patience=5, 
                                            restore_best_weights=True)
    
    # Training with separate validation generator
    model.fit(
        generator()[0],
        epochs=50,                # Max number of epochs
        validation_data = generator()[1],    # Use part of training data for validation
        callbacks=[early_stopping],
        verbose = 1
        )
    
    return model
    
def predict(x_pred, model, df_scaler, target_scaler, features):
    predictions = model.predict(np.array([x[1] for x in x_pred]))
    predictions = target_scaler.inverse_transform(predictions)
    
    predictions = {x_pred[i][0]: float(predictions[i]) for i in range(len(predictions))}

    return predictions


if __name__ == "__main__":
    dataframes = download_data()

    full_df, features = multithread_add_features(dataframes)

    full_df = add_vix(full_df)

    dataframes = full_df.groupby('ticker')
    dataframes = {ticker: df.sort_values(by='date').reset_index(drop=True) for ticker, df in dataframes}

    spy_df = dataframes['SPY']

    del dataframes['SPY']

    train_test_loop(spy_df, dataframes, features)
import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.utils.sp_scraper import SPScraper
import yfinance as yf
from tqdm import tqdm
from multiprocessing import get_context, cpu_count
import pandas as pd
from src.models.model5.TalibIndicators import TalibIndicators
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, BatchNormalization, LeakyReLU, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from copy import deepcopy
import tensorflow as tf
import gc
import matplotlib.pyplot as plt


# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logslogs
tf.get_logger().setLevel('ERROR')         # Suppress TensorFlow INFO logs

scraper = SPScraper()
constituents_df, changes_df, historical_df = scraper.scrape_sp500_symbols()
sp_addition_dates = {ticker: date for ticker, date in zip(constituents_df['ticker_added'].dropna().tolist(), constituents_df.index.dropna().tolist())}

sp_tickers = [ticker.replace(".", "-") for ticker in constituents_df['ticker_added'].dropna().tolist()] + ["^GSPC"]


win_count  = 0
lose_count = 0
wins = []
losses = []
returns = 1
spy_returns = 1

# Global variable to track current plot figure
current_figure = None



temp_data_dir = "src/models/model5/temp_data"
os.makedirs(temp_data_dir, exist_ok=True)

def download_data() -> dict[str, pd.DataFrame]:
    """Download data for all tickers in the S&P 500 and split them up by ticker"""
    dataframes = {}

    # Download data
    data = yf.download(sp_tickers, period="max")

    tickers = data.columns.levels[1]
    for ticker in tqdm(tickers, desc="Splitting data by ticker", unit="ticker"):
        dataframes[ticker] = data.xs(ticker, axis=1, level=1)

    return dataframes

def calculate_z_score():
    pass

def add_features(ticker, df):
    # Make all columns lower case for easier indexing
    df.reset_index(inplace=True)
    df.columns = df.columns.str.lower()

    # Add ticker and return columns
    df['ticker'] = ticker
    df['return_21d'] = df['close'].pct_change(periods=21, fill_method = None).shift(-21)

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
    args = [(ticker, ticker_df) for ticker, ticker_df in dataframes.items()]

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

def drop_data(dataframes, drop_date, pred_date = None):
    """Drops data after specified date (whatever index spy data is on in the loop) and before S&P addition"""
    pred_data = pd.DataFrame()
    for ticker, df in tqdm(dataframes.items(), total=len(dataframes), desc="Dropping data", unit="ticker"):
        # Get test data NOT Y TRAIN
        # Convert sp_addition_dates to the same format as the date column
        if ticker in sp_addition_dates:
            addition_date = pd.to_datetime(sp_addition_dates[ticker]).date()
            dataframes[ticker].drop(df[df['date'] <= addition_date].index, inplace=True)
        if pred_date:
            pred_idx = df.index[df['date'] == pred_date]
            if not pred_idx.empty:
                pred_data = pd.concat([pred_data, df.loc[df['date'] == pred_date]], ignore_index=True)
        dataframes[ticker].drop(df[df['date'] > drop_date].index, inplace=True)
    dropped_dataframes = {ticker: df for ticker, df in dataframes.items() if not df.empty}
    return dropped_dataframes, pred_data

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
    
    label_encoder.fit(sp_tickers)
    df["encoded_ticker"] = df["ticker"].map(lambda x: label_encoder.transform([x])[0] if x in sp_tickers else -1)

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

def plot_predictions_vs_actual(predictions, pred_df, date):
    """Plot predicted vs actual returns for each stock - ordered by predicted returns"""
    # Create lists to store data for plotting
    plot_data = []
    
    # Extract data for plotting
    for ticker, pred_return in predictions.items():
        ticker_data = pred_df[pred_df['ticker'] == ticker]['return_21d']
        if not ticker_data.empty:
            actual_return = ticker_data.iloc[0] * 100  # Convert to percentage
            predicted_return = pred_return * 100  # Convert to percentage
            plot_data.append((ticker, predicted_return, actual_return))
    
    # Sort by predicted returns (ascending)
    plot_data.sort(key=lambda x: x[1])
    
    # Unpack sorted data
    tickers = [item[0] for item in plot_data]
    predicted_returns = [item[1] for item in plot_data]
    actual_returns = [item[2] for item in plot_data]
    
    # Close previous figure if it exists
    global current_figure
    if current_figure is not None:
        plt.close(current_figure)
    
    # Set up the new plot
    current_figure = plt.figure(figsize=(16, 10))
    
    # Create x positions for the stocks
    x_positions = range(len(tickers))
    
    # Plot predicted returns as blue bars
    plt.bar(x_positions, predicted_returns, alpha=0.7, color='steelblue', 
            label='Predicted Returns', width=0.8)
    
    # Plot actual returns as red circles
    plt.scatter(x_positions, actual_returns, color='red', s=60, 
               label='Actual Returns', zorder=5, alpha=0.8)
    
    # Set appropriate y-axis limits with some padding
    all_returns = predicted_returns + actual_returns
    y_min = min(all_returns) - 1
    y_max = max(all_returns) + 1
    plt.ylim(y_min, y_max)
    
    # Customize the plot
    plt.xlabel('Stocks (Ordered by Predicted Returns)', fontsize=12)
    plt.ylabel('Returns (%)', fontsize=12)
    plt.title(f'Predicted vs Actual Returns for {date}', fontsize=14, fontweight='bold')
    
    # Set x-axis ticks and labels
    plt.xticks(x_positions, tickers, rotation=45, ha='right', fontsize=8)
    
    # Add horizontal line at 0%
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
    
    # Add legend
    plt.legend(loc='upper left', fontsize=10)
    
    # Add grid
    plt.grid(True, alpha=0.3, axis='y')
    
    # Calculate and display statistics
    correlation = np.corrcoef(predicted_returns, actual_returns)[0, 1]
    rmse = np.sqrt(np.mean([(p - a)**2 for p, a in zip(predicted_returns, actual_returns)]))
    mae = np.mean([abs(p - a) for p, a in zip(predicted_returns, actual_returns)])
    
    # Add statistics text box
    stats_text = f'Correlation: {correlation:.3f}\nRMSE: {rmse:.2f}%\nMAE: {mae:.2f}%\nStocks: {len(tickers)}'
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Highlight top 10 predictions with annotations
    if len(tickers) >= 10:
        top_10_start = len(tickers) - 10
        for i in range(top_10_start, len(tickers)):
            plt.annotate(f'{actual_returns[i]:.1f}%', 
                        (i, actual_returns[i]), 
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=8, color='darkred',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    # Close any existing plots before showing new one
    plt.show(block=False)  # Non-blocking show
    plt.pause(0.1)  # Small pause to ensure plot is displayed

def train_test_loop(spy_df, dataframes, features):

    for i in range(8316, len(spy_df) - 21, 21):
        # Clear any old sliding-window files so we only train on current window
        for f in os.listdir(temp_data_dir):
            os.remove(os.path.join(temp_data_dir, f))

        #Drop date is y train
        drop_date = spy_df.iloc[i]['date']
        pred_date = spy_df.iloc[i + 21]['date']
        spy_pred = spy_df.iloc[i + 21]['return_21d']

        # Passing a copy of the df so the original df is intact
        dropped_dataframes, pred_df = drop_data(deepcopy(dataframes), drop_date, pred_date)

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
        dropped_dataframes, _ = drop_data(deepcopy(dataframes), pred_drop_date)

        z_score_dataframes, z_score_features = z_score_data(dropped_dataframes, features)

        scaled_dataframe, scaled_features, _, _ = scale_data(z_score_dataframes, z_score_features, df_scaler, target_scaler)

        x_pred = create_pred_sliding_window(scaled_dataframe, scaled_features)

        predictions = predict(x_pred, model, df_scaler, target_scaler, scaled_features)

        del model

        os.system('cls' if os.name == 'nt' else 'clear')    

        # top_10_predictions = dict(sorted(predictions.items(), 
        #                                     key=lambda x: x[1], 
        #                                     reverse=True)[:10])
        top_10_predictions = dict(sorted(predictions.items(), 
                                            key=lambda x: x[1], 
                                            reverse=True))
        
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
        
        # Plot predictions vs actual returns
        plot_predictions_vs_actual(predictions, pred_df, spy_df.iloc[i + 21]['date'])

        global win_count, lose_count, wins, losses, returns

        if avg_real_return > spy_pred:
            win_count += 1
            wins.append(avg_real_return - spy_pred)
        else:
            lose_count += 1
            losses.append(spy_pred - avg_real_return)

        global returns, spy_returns
        returns *= (1+avg_real_return)
        spy_returns *= (1+spy_pred)


        print(f"Win Count: {win_count}, Lose Count: {lose_count}")
        print(f"Average win by {np.mean(wins):.2f}")
        print(f"Average loss by {np.mean(losses):.2f}")
        print(f"Total return {((returns - 1) * 100):.2f}%")
        print(f"Spy returns {((spy_returns-1) * 100):.2f}%")


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
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', 
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

    spy_df = dataframes['^GSPC']

    del dataframes['^GSPC']

    train_test_loop(spy_df, dataframes, features)
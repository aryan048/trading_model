import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sqlalchemy import create_engine, inspect
from tqdm import tqdm

# Database connection
DATABASE_URL = "sqlite:////Users/aryanhazra/Downloads/VSCode Repos/trading_model/src/pre_processing/stock_data/stock_data.db"
engine = create_engine(DATABASE_URL)

# Get all table names (assuming each table is a stock ticker)
def get_stock_tables():
    inspector = inspect(engine)
    return inspector.get_table_names()


# Load stock data from table
def load_stock_data(ticker):
    query = f'SELECT * FROM "{ticker}" ORDER BY date'
    df = pd.read_sql(query, con=engine)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    #these columns should be dropped for all
    df = df.drop(columns=['acos', 'asin'])
    #need to drop NA columns
    df = df.dropna()

    # Select features (close + any technical indicators if available)
    features = [col for col in df.columns if col != 'date']
    return df[features]

def prepare_data(df, ticker_mapping, lookback=30):
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Encode 'ticker' as a numeric feature
    df["ticker_encoded"] = df["ticker"].map(ticker_mapping)
    print(df['ticker_encoded'].unique())

    # Ensure 'close' is the first column in scaled data (important for y selection)
    feature_columns = ['close'] + [col for col in df.columns if col not in ['ticker', 'close']]
    
    # Scale only numeric features
    scaled_features = scaler.fit_transform(df[feature_columns])
    
    # Add the encoded ticker column back as a feature
    scaled_data = np.column_stack((scaled_features, df['ticker_encoded'].values))

    X, y = [], []
    for i in range(len(scaled_data) - lookback - 30):
        X.append(scaled_data[i:i+lookback])  # Use last 30 days as input
        y.append(df['close'].values[i+lookback:i+lookback+30])  # Predict actual close prices

    return np.array(X), np.array(y), scaler


# Define or load LSTM model
def build_or_load_model(input_shape, model_path="lstm_model.h5"):
    try:
        model = load_model(model_path)
        print("Loaded existing model.")
    except:
        print("Creating new model...")
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(30)  # Predict next 30 days of prices
        ])
        model.compile(optimizer='adam', loss='mse')

    return model


# Train model on multiple tickers
def train_model():
    tickers = np.array(get_stock_tables())


    encoder = LabelEncoder()
    encoder.fit(tickers)  # Fit once

    # Save mapping
    ticker_mapping = {ticker: idx for idx, ticker in enumerate(encoder.classes_)}    
    model = None

    for ticker in tqdm(tickers, desc="Training models", unit="ticker"):
        try:
            #might use for debugging
            #print(f"Training on {ticker}...")
            df = load_stock_data(ticker)
            #if df is empty it wont split properly
            if df.empty:
                continue
            X, y, scaler = prepare_data(df, ticker_mapping)

            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            #train data dropps 60 days for look forwards and backwards? so it will be less than df, check if its empty
            if any(arr.size == 0 for arr in [X_train, X_test, y_train, y_test]):
                continue

            if model is None:
                model = build_or_load_model((X.shape[1], X.shape[2]))

            model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), verbose=0)
        except:
            continue

    # Save model after training on all tickers
    model.save("lstm_model.h5")
    print("Model saved.")


# Predict next 30 days for each stock
def predict_next_30_days():
    tickers = np.array(get_stock_tables())


    encoder = LabelEncoder()
    encoder.fit(tickers)  # Fit once

    # Save mapping
    ticker_mapping = {ticker: idx for idx, ticker in enumerate(encoder.classes_)}

    model = load_model("lstm_model.h5")

    predictions = {}
    for ticker in tickers:
        df = load_stock_data(ticker)
        
        if not df.empty:
            X, _, scaler = prepare_data(df)

            last_sequence = X[-1].reshape(1, X.shape[1], X.shape[2])
            predicted_scaled = model.predict(last_sequence)[0]
            predicted_prices = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))

            predictions[ticker] = predicted_prices.flatten()

    return predictions


# Run training & prediction
train_model()
predictions = predict_next_30_days()


# Print predictions
for ticker, prices in predictions.items():
    print(f"{ticker} Next 30 Days Predictions:", prices)



import sys

# Add the absolute path to the project root directory
project_root = "/Users/aryanhazra/Downloads/Github Repos/trading_model"
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
from src.models.utils.sp_scraper import SPScraper
import yfinance as yf
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow import keras
import talib as ta
import time
from IPython.display import clear_output
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
import os
from src.models.model4.utils.create_df import create_df


# Initialize SPScraper
scraper = SPScraper()

model = keras.models.load_model('/Users/aryanhazra/Downloads/Github Repos/trading_model/src/models/model4/model4.keras')

# Load all scalers
df_scaler = joblib.load('/Users/aryanhazra/Downloads/Github Repos/trading_model/src/models/model4/scaler_df.pkl')

# Replace '.' with '-' in ticker symbols, also add SPY as a benchmark
_, pred_grouped_dfs, _, _, feature_cols = create_df(data_len="1y")

#take curr day, find most recent day in each df
today = pd.Timestamp.now().strftime('%Y-%m-%d')
x_pred = []
for ticker, df in pred_grouped_dfs.items():
    most_recent_day = df['date'].max()
    most_recent_index = df[df['date'] == most_recent_day].index[0]

    window = df.iloc[most_recent_index - 60:][feature_cols].values

    x_pred.append([ticker,window])

#predicting based on the second vals of x_test (the windows)
predictions = model.predict(np.array([x[1] for x in x_pred]))

# Current predictions shape is (n_samples, 1)
# Need to create a matrix with same number of columns as what scaler was fit on
predictions_matrix = np.zeros((predictions.shape[0], len(feature_cols) + 1))  # 4 columns for log_return_30d, rsi, encoded_ticker, vix
predictions_matrix[:, 0] = predictions.ravel()  # Put predictions in first column (log_return_30d position)

# Now inverse transform the whole matrix
inverse_transformed = df_scaler.inverse_transform(predictions_matrix)

# Take only the first column which contains our inverse-transformed predictions
predictions = inverse_transformed[:, 0]
# Assigning predictions, and real vals based on ticker
predictions = {x_pred[i][0]: float(predictions[i]) for i in range(len(predictions))}
#sorting the predictions by the first value (the predicted vals)
top_10_predictions = dict(sorted(predictions.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)[:10])

clear_output(wait=True)  # The wait=True parameter prevents flickering
print(f"\nPredictions for date: {most_recent_day}")
print("Top 10 Predicted Returns:")
print("Ticker | Predicted Return")
print("-" * 30)
for ticker, pred in top_10_predictions.items():
# Convert to percentages by multiplying by 100
    print(f"{ticker:6} | {pred*100:13.2f}%")
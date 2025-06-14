import sys

# Add the absolute path to the project root directory
project_root = "/Users/aryanhazra/Downloads/Github Repos/trading_model"
if project_root not in sys.path:
    sys.path.append(project_root)
    
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow import keras
import talib as ta
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from src.models.model4.utils.create_df import create_df
from src.models.utils.sp_scraper import SPScraper

# Initialize SPScraper
scraper = SPScraper()

test_data_from = input(f"Enter train/test split year: " )
# Convert empty string to None
test_data_from = test_data_from if test_data_from.strip() else None

model = keras.models.load_model(f'/Users/aryanhazra/Downloads/Github Repos/trading_model/src/models/model4/{int(test_data_from) - 1}/model4.keras')



# Replace '.' with '-' in ticker symbols, also add SPY as a benchmark
train_grouped_dfs, _, df_scaler, _, feature_cols = create_df(test_data_from = test_data_from)

# Assuming your list of tuples is called ticker_df_list
spy_data = next(df for ticker, df in train_grouped_dfs.items() if ticker == "^GSPC")

# Initalize compound returns
predicted_compound_return = 1.0  # Starting with 1 (100%)
real_compound_return = 1.0
spy_real_compound_return = 1.0

# Allow the option to set the oldest start date dynamically
start_idx = spy_data.index[60]  # Get the index of the earliest date
oldest_start_date = spy_data.iloc[start_idx]['date']  # Get the date at that index
print(f"The oldest available start date is: {oldest_start_date}")

# Make SPY data dates timezone-naive
spy_data['date'] = pd.to_datetime(spy_data['date']).dt.tz_localize(None)

# Initialize performance tracking
yearly_performance = {
    'real': {},
    'spy': {}
}
dates = []
real_returns = []
spy_returns = []

# Create figure and axes
plt.ion()  # Turn on interactive mode
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
fig.suptitle('Trading Model Performance', fontsize=16)

for i in tqdm(range(start_idx, len(spy_data) - 21, 21), desc="Processing SPY data", unit="step"):
    target_date = spy_data.iloc[i]['date']
    x_test = []
    y_real = []
    predictions = {}
    for ticker, df in train_grouped_dfs.items():

        # If the ticker is not SPY and the date of the ticker is greater than the target date, then skip
        if ticker != "^GSPC" and pd.to_datetime(scraper.scrape_sp500_symbols().loc[ticker.replace("-", "."), 'date_added']) > target_date:
            continue
        
        # Ensure target_date is timezone-naive
        target_date = pd.to_datetime(target_date).replace(tzinfo=None)

        # Ensure df['date'] is also timezone-naive
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

        # Now this check will work correctly
        if target_date not in df['date'].values:
            continue

        date_idx = df.index[df['date'] == target_date][0]

        # Check if there are 60 days before and 30 after
        if not (date_idx >= 60 and date_idx + 21 < len(df)):
            continue

        window = df.iloc[date_idx - 60:date_idx][feature_cols].values  # shape (60, num_features)

        x_test.append([ticker, window])

        # Predict the "price in 30 days" from the current i-th index (i.e. day 60 of the window)
        y_real.append(df.iloc[date_idx]['log_return_30d'])

    #predicting based on the second vals of x_test (the windows)
    predictions = model.predict(np.array([x[1] for x in x_test]))

    # Current predictions shape is (n_samples, 1)
    # Need to create a matrix with same number of columns as what scaler was fit on
    predictions_matrix = np.zeros((predictions.shape[0], len(feature_cols) + 1))  # +1 for log_return_30d

    # Since 'scaled_log_return_30d' is no longer in feature_cols, assume it is the first column
    log_return_idx = 0  # Explicitly set to 0 as it is the first column in the scaled data
    predictions_matrix[:, log_return_idx] = predictions.ravel()  # Put predictions in the log return column position

    # Now inverse transform the whole matrix
    inverse_transformed = df_scaler.inverse_transform(predictions_matrix)

    # Take only the log return column which contains our inverse-transformed predictions
    predictions = inverse_transformed[:, log_return_idx]
    # Assigning predictions, and real vals based on ticker
    predictions = {x_test[i][0]: (float(predictions[i]), float(y_real[i])) for i in range(len(predictions))}
    #sorting the predictions by the first value (the predicted vals)
    top_10_predictions = dict(sorted(predictions.items(), 
                                        key=lambda x: x[1][0], 
                                        reverse=True)[:10])
    
    # Clear the terminal and print new output
    os.system('clear')
    print(f"\nPredictions for date: {target_date}")
    print("Top 10 Predicted Returns:")
    print("Ticker | Predicted Return | Actual Return")
    print("-" * 45)
    for ticker, (pred, actual) in top_10_predictions.items():
        # Convert to percentages by multiplying by 100
        print(f"{ticker:6} | {pred*100:13.2f}% | {actual*100:12.2f}%")

    # Convert average returns to percentages
    avg_predicted_return = np.mean([pred for _, (pred, _) in top_10_predictions.items()])
    print("\nAverage Predicted Return for Top 10: {:.2f}%".format(avg_predicted_return*100))

    avg_real_return = np.mean([actual for _, (_, actual) in top_10_predictions.items()])
    print("\nAverage Actual Return for Top 10: {:.2f}%".format(avg_real_return*100))

    # For compound returns, we'll show the total percentage gain/loss
    predicted_compound_return *= np.exp(avg_predicted_return)
    print("Predicted Compound Return: {:.2f}%".format((predicted_compound_return-1)*100))

    real_compound_return *= np.exp(avg_real_return)
    print("Real Compound Return: {:.2f}%".format((real_compound_return-1)*100))

    spy_real_compound_return *= np.exp(spy_data.iloc[i]['log_return_30d'])
    print("SPY Real Compound Return: {:.2f}%".format((spy_real_compound_return-1)*100))

    # Track yearly performance
    current_year = target_date.year
    if current_year not in yearly_performance['real']:
        yearly_performance['real'][current_year] = []
        yearly_performance['spy'][current_year] = []
    
    yearly_performance['real'][current_year].append(avg_real_return)
    yearly_performance['spy'][current_year].append(spy_data.iloc[i]['log_return_30d'])
    
    # Update arrays for plotting
    dates.append(target_date)
    real_returns.append(real_compound_return)
    spy_returns.append(spy_real_compound_return)
    
    # Clear previous plots
    ax1.clear()
    ax2.clear()
    
    # Plot cumulative returns (log scale)
    ax1.plot(dates, real_returns, label='Model Returns', color='blue')
    ax1.plot(dates, spy_returns, label='SPY Returns', color='red')
    ax1.set_yscale('log')
    ax1.set_title('Cumulative Returns (Log Scale)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot yearly performance
    years = sorted(yearly_performance['real'].keys())
    real_yearly_returns = [np.mean(yearly_performance['real'][year]) * 100 for year in years]  # Convert to percentage
    spy_yearly_returns = [np.mean(yearly_performance['spy'][year]) * 100 for year in years]  # Convert to percentage
    
    x = np.arange(len(years))
    width = 0.35
    
    ax2.bar(x - width/2, real_yearly_returns, width, label='Model', color='blue')
    ax2.bar(x + width/2, spy_yearly_returns, width, label='SPY', color='red')
    ax2.set_xticks(x)
    ax2.set_xticklabels(years)
    ax2.set_title('Yearly Average Returns (%)')
    ax2.set_ylabel('Return (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    
    # Rotate x-axis labels for better readability
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Draw the plot
    plt.draw()
    plt.pause(0.1)  # Small pause to allow the plot to update

# Turn off interactive mode to finalize the plot
plt.ioff()

# Show the plot and keep it open until manually closed
plt.show()






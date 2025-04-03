# Imports
from tensorflow import keras 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 
from datetime import datetime
import yfinance as yf
from pytz import timezone
from tensorflow.keras.callbacks import EarlyStopping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



data = yf.Ticker("AAPL").history(period="max")
data.reset_index(inplace=True)
data.columns = data.columns.str.lower()


print(data.head())
print(data.info())
print(data.describe())


# Initial Data Visualization
# Plot 1 - Open and Close Prices of time
plt.figure(figsize=(12,6))
plt.plot(data['date'], data['open'], label="Open",color="blue")
plt.plot(data['date'], data['close'], label="Close",color="red")
plt.title("Open-Close Price over Time")
plt.legend()
# plt.show()

# Plot 2 - Trading Volume (check for outliers)
plt.figure(figsize=(12,6))
plt.plot(data['date'],data['volume'],label="Volume",color="orange")
plt.title("Stock Volume over Time")
# plt.show()


# Drop non-numeric columns
numeric_data = data.select_dtypes(include=["int64","float64"])

# Plot 3 - Check for correlation between features
plt.figure(figsize=(8,6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
# plt.show()


# Convert the Data into Date time then create a date filter
data['date'] = pd.to_datetime(data['date'])

# Define the timezone
ny_tz = timezone("America/New_York")

# Use timezone-aware datetime objects
prediction = data.loc[
    (data['date'] > ny_tz.localize(datetime(2013, 1, 1))) &
    (data['date'] < ny_tz.localize(datetime(2018, 1, 1)))
]

plt.figure(figsize=(12,6))
plt.plot(data['date'], data['close'],color="blue")
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Price over time")


# Prepare for the LSTM Model (Sequential)
stock_close = data.filter(["close"])
dataset = stock_close.values #convert to numpy array
training_data_len = int(np.ceil(len(dataset) * 0.997))

# Preprocessing Stages
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)

training_data = scaled_data[:training_data_len] #95% of all out data

X_train, y_train = [], []


# Create a sliding window for our stock (60 days)
for i in range(60, len(training_data)):
    X_train.append(training_data[i-60:i, 0])
    y_train.append(training_data[i,0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Build the Optimized Model
model = keras.models.Sequential()

# First LSTM Layer
model.add(keras.layers.LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))

# Second LSTM Layer
model.add(keras.layers.LSTM(128, return_sequences=True))

# Third LSTM Layer
model.add(keras.layers.LSTM(64, return_sequences=False))

# Dense Layer
model.add(keras.layers.Dense(256, activation="relu"))

# Dropout Layer
model.add(keras.layers.Dropout(0.4))

# Final Output Layer
model.add(keras.layers.Dense(1))

# Compile the Model
model.compile(optimizer="adam",
              loss="mae",
              metrics=[keras.metrics.RootMeanSquaredError()])

# Model Summary
model.summary()

# Define early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=10,         # Stop if no improvement after 10 epochs
    restore_best_weights=True  # Restore the best weights after stopping
)

# Train the Model
training = model.fit(
    X_train, y_train,
    epochs=1000,          # High number of epochs
    batch_size=32,
    validation_split=0.1, # Use 10% of training data for validation
    callbacks=[early_stopping]
)

# Prep the test data
test_data = scaled_data[training_data_len - 60:]
X_test, y_test = [], dataset[training_data_len:]


for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1 ))

# Make a Prediction
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)


# Plotting data
train = data[:training_data_len]
test =  data[training_data_len:]

test = test.copy()

test['Predictions'] = predictions

plt.figure(figsize=(12,8))
plt.plot(train['date'], train['close'], label="Train (Actual)", color='blue')
plt.plot(test['date'], test['close'], label="Test (Actual)", color='orange')
plt.plot(test['date'], test['Predictions'], label="Predictions", color='red')
plt.title("Our Stock Predictions")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()

# Ensure 'date' column is in datetime format
test['date'] = pd.to_datetime(test['date'])

# Filter the test data for 2025
test_2025 = test[test['date'].dt.year == 2025]

# Plotting data for 2025
plt.figure(figsize=(12, 8))
plt.plot(test_2025['date'], test_2025['close'], label="Test (Actual)", color='orange')
plt.plot(test_2025['date'], test_2025['Predictions'], label="Predictions", color='red')
plt.title("Our Stock Predictions for 2025")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()

# Predict an extra 30 days into the future
future_predictions = []
last_60_days = X_test[-1]  # Get the last 60 days of data from the test set

for _ in range(30):  # Predict for the next 30 days
    # Predict the next day
    next_day_prediction = model.predict(last_60_days.reshape(1, last_60_days.shape[0], 1))
    
    # Inverse transform the prediction to get the actual value
    next_day_prediction = scaler.inverse_transform(next_day_prediction)
    
    # Append the prediction to the future_predictions list
    future_predictions.append(next_day_prediction[0, 0])
    
    # Update the last_60_days array by removing the first value and appending the new prediction
    next_day_scaled = scaler.transform([[next_day_prediction[0, 0]]])  # Scale the prediction
    last_60_days = np.append(last_60_days[1:], next_day_scaled)  # Update the sliding window

# Create a DataFrame for the future predictions
future_dates = pd.date_range(start=test['date'].iloc[-1] + pd.Timedelta(days=1), periods=30)
future_df = pd.DataFrame({'date': future_dates, 'Predictions': future_predictions})

# Plot the extended predictions
plt.figure(figsize=(12, 8))
plt.plot(test['date'], test['close'], label="Test (Actual)", color='orange')
plt.plot(test['date'], test['Predictions'], label="Test (Predictions)", color='red')
plt.plot(future_df['date'], future_df['Predictions'], label="Future Predictions (30 Days)", color='green')
plt.title("Stock Predictions with 30-Day Forecast")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()
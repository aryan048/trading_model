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
from datetime import timedelta
from tensorflow.keras.callbacks import EarlyStopping
import optuna

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


best_hyperparams = {'lstm_units_1': 448, 
                        'lstm_units_2': 512, 
                        'dense_units': 256, 
                        'dropout_rate': 0.5863386331655234, 
                        'learning_rate': 0.0003511266706629626, 
                        'optimizer': 'rmsprop'}



data = yf.Ticker("AAPL").history(period="max")
data.reset_index(inplace=True)
data.columns = data.columns.str.lower()
print(data.head())
print(data.info())
print(data.describe())

# Define the split date
split_date = "2025-03-01"  # Replace with your desired date

# Find the index where the date matches the split_date
training_data_len = data[data['date'] < split_date].shape[0]

# Prepare for the LSTM Model (Sequential)
stock_close = data.filter(["close"])
dataset = stock_close.values #convert to numpy array

# Preprocessing Stages
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)

training_data = scaled_data[:training_data_len] #95% of all out data

X_train, y_train = [], []


# Create a sliding window for our stock (60 days)
# Use the past 60 days to predict the price 30 days out
for i in range(60, len(training_data)-30):
    X_train.append(training_data[i-60:i, 0])
    y_train.append(training_data[i+30,0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Best hyperparameters: {'lstm_units_1': 448, 'lstm_units_2': 512, 'dense_units': 256, 'dropout_rate': 0.5863386331655234, 'learning_rate': 0.0003511266706629626, 'optimizer': 'rmsprop'}
# Build the model using the best hyperparameters
model = keras.models.Sequential()
model.add(keras.layers.LSTM(best_hyperparams["lstm_units_1"], return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(keras.layers.LSTM(best_hyperparams["lstm_units_2"], return_sequences=False))
model.add(keras.layers.Dense(best_hyperparams["dense_units"], activation="relu"))
model.add(keras.layers.Dropout(best_hyperparams["dropout_rate"]))
model.add(keras.layers.Dense(1))

# Compile model
optimizer = keras.optimizers.Adam(learning_rate=best_hyperparams["learning_rate"]) if best_hyperparams["optimizer"] == 'adam' else keras.optimizers.RMSprop(learning_rate=best_hyperparams["learning_rate"])
model.compile(optimizer=optimizer, loss="mae", metrics=[keras.metrics.RootMeanSquaredError()])

# Train the final model with the best hyperparameters
early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# Train the model with as many epochs as possible
training = model.fit(X_train, y_train, epochs=1000, batch_size=32, callbacks=[early_stopping])

# Prep the test data
# Scaled data is all closing prices scaled from 0-1
# Start predicting from training_data_len onwards, but needs 60 days prior to do that
test_data = scaled_data[training_data_len - 60:]
# X__test is empty, and y_text will be the actual prices 30 days out onwards, this var isnt acc used
X_test, y_test = [], dataset[training_data_len+30:]


# Create a sliding window for our stock (60 days)
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

plt.autoscale()


# Prompt the user to input a date for the x-axis limit
manual_date = input("Enter a start date for the plot (YYYY-MM-DD): ")

# Convert the input date to a datetime object
try:
    manual_date = pd.to_datetime(manual_date)
except ValueError:
    print("Invalid date format. Defaulting to one year ago.")
    manual_date = data['date'].iloc[-1] - timedelta(days=365)

# Set the x-axis limits for the plot
plt.xlim([manual_date, data['date'].iloc[-1]])
plt.show()
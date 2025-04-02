# Imports
from tensorflow import keras 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 
from datetime import datetime
from sqlalchemy import create_engine, inspect
from tqdm import tqdm
from tensorflow.keras import backend as K
from pympler import muppy, summary, tracker
import sys
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


DATABASE_URL = "sqlite:////Users/aryanhazra/Downloads/VSCode Repos/trading_model/src/pre_processing/stock_data/stock_data.db"
engine = create_engine(DATABASE_URL)
inspector = inspect(engine)
tables = inspector.get_table_names()


# Initialize the model once (outside the loop)
model = keras.models.Sequential()

# Define the model structure (same as before)
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(60, 1)))
model.add(keras.layers.LSTM(64, return_sequences=False))
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))

model.compile(optimizer="adam", loss="mae", metrics=[keras.metrics.RootMeanSquaredError()])

# Check for saved model weights (you can also store by ticker if you prefer)
model_weights_file = "my_model.weights.h5"

for ticker in tqdm(tables, desc="Processing Tickers"):
    data = pd.read_sql(f'SELECT * FROM "{ticker}"', con=engine)

    # Drop non-numeric columns
    numeric_data = data.select_dtypes(include=["int64","float64"])

    # Convert the Data into Date time then create a date filter
    data['date'] = pd.to_datetime(data['date'])

    prediction = data.loc[
        (data['date'] > datetime(2020,1,1)) &
        (data['date'] < datetime(2025,1,1))
    ]

    # Prepare for the LSTM Model (Sequential)
    stock_close = data.filter(["close"])
    dataset = stock_close.values #convert to numpy array
    training_data_len = int(np.ceil(len(dataset) * 0.95))

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

    if len(X_train) <= 1:
        continue


    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    if os.path.exists(model_weights_file):
        model.load_weights(model_weights_file)

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=0)
    model.save_weights("my_model.weights.h5")

    #move out of function soon
    # Prep the test data
    test_data = scaled_data[training_data_len - 60:]
    X_test, y_test = [], dataset[training_data_len:]


    for i in range(60, len(test_data)):
        X_test.append(test_data[i-60:i, 0])
        
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1 ))

    model.save_weights(model_weights_file)

    # Make a Prediction
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)


    # Plotting data
    train = data[:training_data_len]
    test =  data[training_data_len:]

    test = test.copy()

    test['Predictions'] = predictions

    # plt.figure(figsize=(12,8))
    # plt.plot(train['date'], train['close'], label="Train (Actual)", color='blue')
    # plt.plot(test['date'], test['close'], label="Test (Actual)", color='orange')
    # plt.plot(test['date'], test['Predictions'], label="Predictions", color='red')
    # plt.title(f"{ticker} Stock Predictions")
    # plt.xlabel("Date")
    # plt.ylabel("Close Price")
    # plt.legend()
    # plt.show(block=False)  # Non-blocking plot
    # plt.pause(3)  # Keep open for 3 seconds
    # plt.close()  # Close the figure to free memory

    K.clear_session()  # Clear the session to free up resources
    gc.collect()  # Force garbage collection


    all_objects = muppy.get_objects()
    summary.print_(summary.summarize(all_objects))
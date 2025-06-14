# Next steps:
# Make requests in batch
# optimize df creation speed
# Add financiclas
import sys
import os

# Add the absolute path to the project root directory
project_root = "/Users/aryanhazra/Downloads/Github Repos/trading_model"
if project_root not in sys.path:
    sys.path.append(project_root)

from tqdm import tqdm
import numpy as np
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, BatchNormalization, LeakyReLU, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
import joblib
from src.models.model4.utils.create_df import create_df
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import shutil
import tensorflow as tf
import math
import logging

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
tf.get_logger().setLevel('ERROR')         # Suppress TensorFlow INFO logs

tf.get_logger().setLevel(logging.ERROR)


# Directory to save intermediate training data
temp_data_dir = "src/models/model4/temp_data"
os.makedirs(temp_data_dir, exist_ok=True)

train_data_till = input(f"Enter train/test split year: " )
# Convert empty string to None
train_data_till = train_data_till if train_data_till.strip() else None

# Can pass through time period to create_df "1y"
train_grouped_dfs, _, df_scaler, label_encoder, feature_cols = create_df(train_data_till = train_data_till)


# Save scalers
output_dir = f"src/models/model4/{train_data_till}"
os.makedirs(output_dir, exist_ok=True)
joblib.dump(df_scaler, f"src/models/model4/{train_data_till}/scaler_df.pkl")
joblib.dump(label_encoder, f"src/models/model4/{train_data_till}/label_encoder.pkl")

# List the features you want to include (excluding 'price in 30 days' and 'date')
# scaled_close, scaled_sma's are scaled by ticker, rest by total df

def process_ticker(ticker, df, feature_cols, validation_split=0.1):
    local_x_train, local_y_train = [], []
    local_x_val, local_y_val = [], []

    if len(df) < 81:
        return None  # Skip saving if data is insufficient

    split_index = int(len(df) * (1 - validation_split))

    # Split the data into training and validation sets
    train_df = df[:split_index]
    val_df = df[split_index:]

    # Process training data
    for i in range(60, len(train_df)):
        window = train_df.iloc[i - 60:i][feature_cols].values  # shape (60, num_features)
        local_x_train.append(window)
        local_y_train.append(train_df.iloc[i]['scaled_log_return_30d'])

    # Process validation data
    for i in range(60, len(val_df)):
        window = val_df.iloc[i - 60:i][feature_cols].values  # shape (60, num_features)
        local_x_val.append(window)
        local_y_val.append(val_df.iloc[i]['scaled_log_return_30d'])

    # Save training and validation data to disk
    np.save(os.path.join(temp_data_dir, f"{ticker}_train_x.npy"), np.array(local_x_train))
    np.save(os.path.join(temp_data_dir, f"{ticker}_train_y.npy"), np.array(local_y_train))
    np.save(os.path.join(temp_data_dir, f"{ticker}_val_x.npy"), np.array(local_x_val))
    np.save(os.path.join(temp_data_dir, f"{ticker}_val_y.npy"), np.array(local_y_val))

    return ticker  # Return the ticker name for tracking

# Multithreading
with ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(process_ticker, ticker, df, feature_cols)
        for ticker, df in train_grouped_dfs.items()
    ]
    for future in tqdm(futures, desc="Processing tickers", unit="ticker"):
        future.result()  # Ensure all tasks are completed

# Clear some memory
del train_grouped_dfs

# Updated data generator to handle new file structure
def data_generator_from_storage_split(temp_data_dir, batch_size):
    train_x_files = sorted([f for f in os.listdir(temp_data_dir) if f.endswith("_train_x.npy")])
    train_y_files = sorted([f for f in os.listdir(temp_data_dir) if f.endswith("_train_y.npy")])
    val_x_files = sorted([f for f in os.listdir(temp_data_dir) if f.endswith("_val_x.npy")])
    val_y_files = sorted([f for f in os.listdir(temp_data_dir) if f.endswith("_val_y.npy")])

    return train_x_files, train_y_files, val_x_files, val_y_files

train_x_files, train_y_files, val_x_files, val_y_files = data_generator_from_storage_split(temp_data_dir, batch_size=512)

def tf_data_generator(x_files, y_files, batch_size):
    def generator():
        for x_file, y_file in zip(x_files, y_files):
            x_data = np.load(os.path.join(temp_data_dir, x_file))
            y_data = np.load(os.path.join(temp_data_dir, y_file))
            for i in range(0, len(x_data), batch_size):
                x_batch = x_data[i:i+batch_size]
                y_batch = y_data[i:i+batch_size]
                if len(x_batch) == 0 or len(y_batch) == 0:
                    continue  # Skip empty batches
                yield x_batch, y_batch

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 60, len(feature_cols)), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    )
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

train_gen = tf_data_generator(train_x_files, train_y_files, batch_size=512)
val_gen = tf_data_generator(val_x_files, val_y_files, batch_size=512)

model = Sequential()

# Convolutional layers for local pattern extraction
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(60, len(feature_cols))))
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
    optimizer=keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.0),
    loss=keras.losses.Huber(delta=1.0),  # Huber = better for stability on noisy targets
    metrics=[keras.metrics.RootMeanSquaredError()]
)


lr_schedule = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                factor=0.5,  
                                                patience=3, 
                                                verbose=1)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                           patience=10, 
                                           restore_best_weights=True)

# Training with separate validation generator
training = model.fit(
    train_gen,
    epochs=200,                # Max number of epochs
    validation_data = val_gen  ,    # Use part of training data for validation
    callbacks=[early_stopping]
    )

# Clean up temporary files after training
shutil.rmtree(temp_data_dir)

model.save(f"src/models/model4/{train_data_till}/model4.keras")
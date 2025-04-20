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


train_data_till = input(f"Enter train/test split year: " )
# Convert empty string to None
train_data_till = train_data_till if train_data_till.strip() else None

# Can pass through time period to create_df "1y"
train_grouped_dfs, _, df_scaler, label_encoder, feature_cols = create_df(train_data_till = train_data_till)


# Save scalers
joblib.dump(df_scaler, "src/models/model4/scaler_df.pkl")
joblib.dump(label_encoder, "src/models/model4/label_encoder.pkl")

# List the features you want to include (excluding 'price in 30 days' and 'date')
# scaled_close, scaled_sma's are scaled by ticker, rest by total df

x_train, y_train = [], []

for ticker, df in tqdm(train_grouped_dfs.items(), desc="Creating sliding windows", unit="ticker"):

    if len(df) < 81:
        continue
    
    for i in range(60, len(df) - 21):
        # Extract a sliding window of all desired features
        window = df.iloc[i - 60:i][feature_cols].values  # shape (60, num_features)

        # Optional: add ticker as a numeric value if it's useful
        # ticker_id = your_ticker_encoding[ticker]  # if you're using one-hot or label encoding
        # ticker_column = np.full((60, 1), ticker_id)
        # window = np.hstack((window, ticker_column))

        x_train.append(window)

        # Predict the "price in 30 days" from the current i-th index (i.e. day 60 of the window)
        y_train.append(df.iloc[i]['scaled_log_return_30d'])

# Clear some memory
del train_grouped_dfs
        
# Convert to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# For use in colab
np.save('src/models/model4/x_train.npy', x_train)
np.save('src/models/model4/y_train.npy', y_train)


model = Sequential()

# Convolutional layers for local pattern extraction
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
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


lr_schedule = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                factor=0.5, 
                                                patience=3, 
                                                verbose=1)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                           patience=10, 
                                           restore_best_weights=True)

# Model training params for colab
training = model.fit(
    x_train, y_train,
    epochs=200,                # Max number of epochs
    batch_size=512,
    validation_split=0.01,      # Use part of training data for validation
    callbacks=[early_stopping]
)
model.save("src/models/model4/model4.keras")
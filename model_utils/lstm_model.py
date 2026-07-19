import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ============================================================= #
######################### DATA FETCHING #########################
# ============================================================= #

def fetch_traffic_data():
    """
    Fetches all CRZ Traffic Data from NY Gov API in chunks
    """

    url = "https://data.ny.gov/resource/t6yz-b64h.json"

    chunk_size = 50000

    chunks = []

    offset = 0

    while True:
        print(f"Fetching rows {offset + 1}-{offset+chunk_size} ...")

        params = {
            "$limit": chunk_size,
            "$offset": offset
        }

        response = requests.get(url,params=params)

        data = response.json()

        if len(data)==0:

            print("No more data to fetch.")
            break

        chunks.append(pd.DataFrame(data))
        offset += chunk_size

    df_all = pd.concat(chunks, ignore_index=True)

    # Convert date columns
    df_all['toll_day'] = pd.to_datetime(df_all['toll_day'])
    df_all['hour_of_day'] = pd.to_numeric(df_all['hour_of_day'])
    df_all['minute_of_hour'] = pd.to_number(df_all['minute_of_hour'])
    df_all['traffic_volume'] = pd.to_numeric(df_all['traffic_volume'])

    return df_all

# ============================================================= #
###################### DATA PREPROCESSING #######################
# ============================================================= #

def prepare_data(df_all, detection_region=None):
    """
    Prepare data for LSTM modeling

    Parameters:
    - df_all: full dataframe
    - detection_region: specific region or None for all regions

    Returns:
    - data: aggregated timeseries dataframe
    """

    highest_day = df_all['toll_day'].max()

    removal_date = highest_day - timedelta(days=14)

    # Filter data
    data = df_all[df_all['toll_day'] <= removal_date].copy()

    if detection_region:
        data = data[data['detection_region'] == detection_region]

    # Select relevant columns and group
    data = data[['toll_day','hour_of_day', 'minute_of_hour', 'traffic_volume']].copy()

    # First aggregation: Group by time components
    data = data.groupby(['toll_day',' hour_of_day', 'minute_of_hour']).agg(
        {'traffic_volume':'sum'}
    ).reset_index()

    data = data.sort_values('toll_timestamp').reset_index(drop=True)

    return data

# ============================================================= #
############### LSTM SEQUENCES & MODEL TRAINING #################
# ============================================================= #

def make_sequences(arr_2d, lookback=48):
    """
    Create sequences for LSTM
    """
    n = arr_2d.shape[0]

    X = np.zeros((n - lookback, lookback, arr_2d.shape[1]), dtype=np.float32)
    
    y = np.zeros((n - lookback, arr_2d.shape[1]), dtype=np.float32)

    for i in range(n - lookback):

        X[i] = arr_2d[i:i + lookback]
        
        y[i] = arr_2d[i + lookback]

    return X, y

def train_lstm_model(X_train, y_train, X_val, y_val, epochs=50):
    """
    Build and train LSTM model
    """
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]),
             return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val,y_val),
        epochs=epochs,
        batch_size=32,
        verbose=0
    )

    return model, history

def predict_and_evaluate(model, X_val, y_val, scaler):
    """
    Make predictions and calculate metrics
    """
    y_pred_scaled = model.predict(X_val, verbose=0)

    # Inverse transform to original scaling
    y_val_orig = scaler.inverse_transform(y_val.reshape(-1,1)).flatten()

    y_pred_orig = scaler.inverse_transform(y_pred_scaled).flatten()

    # Calcualte metrics

    rmse = np.sqrt(np.mean((y_val_orig - y_pred_orig) ** 2))

    mae = np.mean(np.abs(y_val_orig - y_pred_orig))

    mean_volume = y_val_orig.mean()

    rmse_pct = (rmse / mean_volume) * 100

    mae_pct = (mae / mean_volume) * 100

    return y_val_orig, y_pred_orig, rmse_pct, mae_pct

# ============================================================= #
########################### PIPELINE ############################
# ============================================================= #

def run_model_pipeline(detection_region=None):
    """
    Run complete LSTM pipeline for a given detection region.

    Parameters:
    - detection_region: specific region (e.g., 'Brooklyn') or None for all regions

    Returns:
    - model: trained Keras LSTM model
    - scaler: fitted MinMaxScaler
    - data: prepared Time series data
    - metrics: dictionary with RMSE % and MAE %
    - predictions: dict with actual and predicted values
    """

    # Fetch and prepare data
    df_all = fetch_traffic_data()

    print(f"Preparing data for {detection_region if detection_region else 'all regions'}...")
    
    data = prepare_data(df_all, detection_region)

    # Scale data
    scaler = MinMaxScaler()

    y_scaled = scaler.fit_transform(data[['traffic_volume']].values)

    # Create sequences
    LOOKBACK = 48 # 48 half-hours = 24 hours
    X, y = make_sequences(y_scaled, LOOKBACK)

    # Train/val split (80/20)
    split = int(0.8 * len(X))

    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Train model
    print('Training LSTM Model...')
    
    model, history = train_lstm_model(X_train, y_train, X_val, y_val, epochs=50)

    # Evaluate
    y_val_orig, y_pred_orig, rmse_pct, mae_pct = predict_and_evaluate(
        model, X_val, y_val, scaler
    )

    metrics = {
        'rmse_pct':rmse_pct,
        'mae_pct':mae_pct,
        'mean_volume':y_val_orig.mean()
    }

    predictions = {
        'actual':y_val_orig,
        'predicted':y_pred_orig
    }

    return model, scaler, data, metrics, predictions
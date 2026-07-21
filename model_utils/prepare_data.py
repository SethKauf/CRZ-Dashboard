import requests
import pandas as pd
import numpy as np
import holidays
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt



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

    return df_all

# ============================================================= #
###################### DATA PREPROCESSING #######################
# ============================================================= #

def prepare_data(df, detection_region=None):
    """
    Prepare data for LSTM modeling

    Parameters:
    - df_all: full dataframe
    - detection_region: specific region or None for all regions

    Returns:
    - data: aggregated timeseries dataframe
    """

    # Convert date columns
    df['toll_date'] = pd.to_datetime(df['toll_date'])
    df['hour_of_day'] = pd.to_numeric(df['hour_of_day'])
    df['minute_of_hour'] = pd.to_numeric(df['minute_of_hour'])
    df['crz_entries'] = pd.to_numeric(df['crz_entries'])


    highest_day = df['toll_date'].max()

    removal_date = highest_day - timedelta(days=14)

    # Filter data
    data = df[df['toll_date'] <= removal_date].copy()

    if detection_region:
        data = data[data['detection_region'] == detection_region]

    # Select relevant columns and group
    data = data[['toll_date','hour_of_day', 'minute_of_hour', 'crz_entries']].copy()

    # First aggregation: Group by time components
    data = data.groupby(['toll_date',' hour_of_day', 'minute_of_hour']).agg(
        {'crz_entries':'sum'}
    ).reset_index()

    data = data.sort_values('toll_10_minute_block').reset_index(drop=True)

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
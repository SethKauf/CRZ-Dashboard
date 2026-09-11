import json
import joblib
import holidays
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import load_model

from src.mappings import GROUP_TO_REGION
from src.model_functions import create_future_features

# set paths
DATA_PATH = "data/modeling_data.csv"
MODEL_PATH = "models/lstm_traffic_model.keras"
SCALER_PATH = "models/traffic_scaler.joblib"
CONFIG_PATH = "src/model_config.json"

FORECAST_PATH = "data/forecasts/traffic_forecast.csv"

# model features
NUMERIC_FEATURES = [
    "traffic_scaled",
    "holiday_ind",
    "overnight_ind",
    "dow_sin",
    "dow_cos"
]

# config file
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

LAST_TRAINED_TIMESTAMP = pd.Timestamp(
    config["last_trained_timestamp"]
)

SEQ_LENGTH = config["seq_length"]

FORECAST_DAYS = 15

forecast_start = (
    LAST_TRAINED_TIMESTAMP
    + pd.Timedelta(minutes=10)
)

forecast_end = (
    forecast_start
    + pd.Timedelta(days=FORECAST_DAYS)
    - pd.Timedelta(minutes=10)
)

# load model and scaler
model = load_model(
    MODEL_PATH,
    compile=False
)

traffic_scaler = joblib.load(
    SCALER_PATH
)

# load historical data
data = pd.read_csv(
    DATA_PATH
)

data["toll_10_minute_block"] = pd.to_datetime(
    data["toll_10_minute_block"]
)

data = (
    data
    .sort_values(
        [
            "group_id",
            "toll_10_minute_block"
        ]
    )
    .reset_index(drop=True
    )
)

# Ensure model has been trained on data
data = data[
    data["toll_10_minute_block"]
    <= LAST_TRAINED_TIMESTAMP
].copy()

# scale historical traffic data
data["traffic_scaled"] = (
    traffic_scaler
    .transform(
        data[["traffic_volume"]]
    )
    .ravel()
)

# feature helpers
us_holidays = holidays.US()

# overnight classification since future data wouldn't have that as a feature the same way the MTA data does
data["clock_time"] = (
    data["toll_10_minute_block"]
    .dt.strftime("%H:%M")
)

overnight_lookup = (
    data[
        [
            "clock_time",
            "overnight_ind"
        ]
    ]
    .drop_duplicates()
    .set_index("clock_time")["overnight_ind"]
    .to_dict()
)

# build 144-step history for each group

histories = {}

group_ids = sorted(
    GROUP_TO_REGION.keys()
)

for group_id in group_ids:

    group_history = (
        data[
            data["group_id"] == group_id
        ]
        .sort_values(
            "toll_10_minute_block"
        )
        .tail(
            SEQ_LENGTH
        )
        .copy()
    )

    if len(group_history) < SEQ_LENGTH:
        raise ValueError(
            f"Group {group_id} has only {len(group_history)} historical rows;\n\nNeed {SEQ_LENGTH}"
        )

    histories[group_id] = (
        group_history[
            [
            "toll_10_minute_block",
            *NUMERIC_FEATURES
            ]
        ]
        .copy()
    )

# create future timestamps
future_times = pd.date_range(
    start=forecast_start,
    end=forecast_end,
    freq="10min"
)

print(f"""
Forecasting {len(future_times):,} 10 minute intervals per group
""")

# Recursive Forecast
forecast_rows = []

for step, timestamp in enumerate(future_times, start=1):

    numeric_batch = []
    region_batch = []
    group_batch = []

    # model inputs for all 12 groups
    for group_id in group_ids:

        history = (
            histories[group_id]
            .tail(SEQ_LENGTH)
        )

        numeric_batch.append(
            history[
                NUMERIC_FEATURES
            ].to_numpy(
                dtype=np.float32
            )
        )

        region_id = GROUP_TO_REGION[
            group_id
        ]

        region_batch.append(
            np.full(
                SEQ_LENGTH,
                region_id,
                dtype=np.int32
            )
        )

        group_batch.append(
            np.full(
                SEQ_LENGTH,
                group_id,
                dtype=np.int32
            )
        )

    numeric_batch = np.stack(
        numeric_batch
    )

    region_batch = np.stack(
        region_batch
    )

    group_batch = np.stack(
        group_batch
    )

    # predict all 12 groups simultaneously
    predictions_scaled = (
        model.predict(
            {
                "numeric":numeric_batch,
                "region":region_batch,
                "group":group_batch
            },
            verbose=0
        )
        .reshape(-1)
    )

    # convert preds back to actual counts (un-scale)
    predictions_actual = (
        traffic_scaler
        .inverse_transform(
            predictions_scaled.reshape(
                -1,
                1
            )
        )
        .ravel()
    )

    # calendar features are identical across groups
    future_features = (
        create_future_features(
            timestamp, us_holidays=us_holidays, overnight_lookup=overnight_lookup
        )
    )

    # add predicted row back into each group's history
    for i, group_id in enumerate(
        group_ids
    ):
        region_id = GROUP_TO_REGION[
            group_id
        ]

        new_history_row = {
            **future_features,

            "traffic_scaled":
                predictions_scaled[i]
        }

        histories[group_id] = pd.concat(
            [
                histories[group_id],
                pd.DataFrame(
                    [new_history_row]
                )
            ],
            ignore_index=True
        )

        forecast_rows.append(
            {
                "toll_10_minute_block":
                    timestamp,

                "region_id":
                    region_id,

                "group_id":
                    group_id,

                "predicted_volume":
                    float(
                        predictions_actual[i]
                    )
            }
        )

    # progress update after all 12 groups are processed
    if step % 100 == 0 or step == len(future_times):
        print(
            f"Forecasted {step:,}/{len(future_times):,} "
            f"intervals through {timestamp}"
        )

# save forecast
forecast_df = pd.DataFrame(
    forecast_rows
)

forecast_df.to_csv(
    FORECAST_PATH,
    index=False
)

print()
print(
    f"""
    Saved {len(forecast_df):,} predictions to {FORECAST_PATH}
""" 
)

print()
print(forecast_df.head())

print()
print(forecast_df.tail())

archive_path = (
    f"data/forecasts/archive/traffic_forecast_{LAST_TRAINED_TIMESTAMP.strftime('%Y%m%d')}.csv"
)

forecast_df.to_csv(
    archive_path,
    index=False
)

print(f"Saved archive forecast file: {archive_path}")
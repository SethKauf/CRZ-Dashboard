import os

# Must be set before importing TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import warnings
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import holidays
import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model

from src.mappings import GROUP_TO_REGION
from src.model_functions import create_future_features


warnings.filterwarnings("ignore")


# ============================================================
# PATHS / CONSTANTS
# ============================================================

DATA_PATH = "data/modeling_data.csv"
MODEL_PATH = "models/lstm_traffic_model.keras"
SCALER_PATH = "models/traffic_scaler.joblib"
CONFIG_PATH = "src/model_config.json"

FORECAST_PATH = "data/forecasts/traffic_forecast.csv"
FORECAST_ARCHIVE_DIR = "data/forecasts/archive"

FORECAST_DAYS = 15


NUMERIC_FEATURES = [
    "traffic_scaled",
    "holiday_ind",
    "overnight_ind",
    "dow_sin",
    "dow_cos"
]


# ============================================================
# ROUNDING HELPER
# ============================================================

def round_half_up(value):
    """
    Round using the conventional 0.5-up rule.

    Examples:
        40.49 -> 40
        40.50 -> 41
        40.51 -> 41
    """

    return int(
        Decimal(str(value)).quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP
        )
    )


# ============================================================
# FORECAST PIPELINE
# ============================================================

def run_forecast_generation(
    data_path=DATA_PATH,
    model_path=MODEL_PATH,
    scaler_path=SCALER_PATH,
    config_path=CONFIG_PATH,
    forecast_path=FORECAST_PATH,
    forecast_archive_dir=FORECAST_ARCHIVE_DIR,
    forecast_days=FORECAST_DAYS
):
    """
    Generate recursive 10-minute traffic forecasts for all groups.

    Forecasts begin 10 minutes after the model's
    last_trained_timestamp and continue for forecast_days.

    Returns
    -------
    dict
        Forecast dataframe and output metadata.
    """

    # ========================================================
    # LOAD CONFIG
    # ========================================================

    with open(config_path, "r") as f:
        config = json.load(f)

    last_trained_timestamp = pd.Timestamp(
        config["last_trained_timestamp"]
    )

    seq_length = config["seq_length"]


    # ========================================================
    # DEFINE FORECAST RANGE
    # ========================================================

    forecast_start = (
        last_trained_timestamp
        + pd.Timedelta(minutes=10)
    )

    forecast_end = (
        forecast_start
        + pd.Timedelta(days=forecast_days)
        - pd.Timedelta(minutes=10)
    )

    print(
        "\nForecast start:",
        forecast_start
    )

    print(
        "Forecast end:",
        forecast_end
    )


    # ========================================================
    # LOAD MODEL + SCALER
    # ========================================================

    model = load_model(
        model_path,
        compile=False
    )

    traffic_scaler = joblib.load(
        scaler_path
    )


    # ========================================================
    # LOAD HISTORICAL DATA
    # ========================================================

    data = pd.read_csv(
        data_path
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
        .reset_index(drop=True)
    )


    # Only use observations the model has actually trained on
    data = data[
        data["toll_10_minute_block"]
        <= last_trained_timestamp
    ].copy()


    # ========================================================
    # SCALE HISTORICAL TRAFFIC
    # ========================================================

    data["traffic_scaled"] = (
        traffic_scaler
        .transform(
            data[["traffic_volume"]]
        )
        .ravel()
    )


    # ========================================================
    # FUTURE FEATURE HELPERS
    # ========================================================

    us_holidays = holidays.US()

    # Determine which 10-minute clock times are considered
    # "Overnight" based on the historical dataset.
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
        .set_index(
            "clock_time"
        )["overnight_ind"]
        .to_dict()
    )


    # ========================================================
    # BUILD INITIAL 144-STEP HISTORY FOR EACH GROUP
    # ========================================================

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
                seq_length
            )
            .copy()
        )

        if len(group_history) < seq_length:

            raise ValueError(
                f"Group {group_id} has only "
                f"{len(group_history)} historical rows; "
                f"need {seq_length}."
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


    # ========================================================
    # CREATE FUTURE TIMESTAMPS
    # ========================================================

    future_times = pd.date_range(
        start=forecast_start,
        end=forecast_end,
        freq="10min"
    )

    print(
        f"\nForecasting "
        f"{len(future_times):,} "
        f"10-minute intervals per group.\n"
    )


    # ========================================================
    # RECURSIVE FORECAST
    # ========================================================

    forecast_rows = []


    for step, timestamp in enumerate(
        future_times,
        start=1
    ):

        numeric_batch = []
        region_batch = []
        group_batch = []


        # ----------------------------------------------------
        # MODEL INPUTS FOR ALL 12 GROUPS
        # ----------------------------------------------------

        for group_id in group_ids:

            history = (
                histories[group_id]
                .tail(seq_length)
            )

            numeric_batch.append(
                history[
                    NUMERIC_FEATURES
                ]
                .to_numpy(
                    dtype=np.float32
                )
            )


            region_id = (
                GROUP_TO_REGION[
                    group_id
                ]
            )


            region_batch.append(
                np.full(
                    seq_length,
                    region_id,
                    dtype=np.int32
                )
            )


            group_batch.append(
                np.full(
                    seq_length,
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


        # ----------------------------------------------------
        # PREDICT ALL 12 GROUPS SIMULTANEOUSLY
        # ----------------------------------------------------

        predictions_scaled = (
            model.predict(
                {
                    "numeric":
                        numeric_batch,

                    "region":
                        region_batch,

                    "group":
                        group_batch
                },
                verbose=0
            )
            .reshape(-1)
        )


        # ----------------------------------------------------
        # CONVERT BACK TO ACTUAL TRAFFIC COUNTS
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # FUTURE CALENDAR FEATURES
        # ----------------------------------------------------

        future_features = (
            create_future_features(
                timestamp,
                us_holidays=us_holidays,
                overnight_lookup=overnight_lookup
            )
        )


        # ----------------------------------------------------
        # UPDATE EACH GROUP'S HISTORY
        # ----------------------------------------------------

        for i, group_id in enumerate(
            group_ids
        ):

            region_id = (
                GROUP_TO_REGION[
                    group_id
                ]
            )

            predicted_volume = float(
                predictions_actual[i]
            )


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
                        predicted_volume,

                    "predicted_volume_rounded":
                        round_half_up(
                            predicted_volume
                        )
                }
            )


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            step % 100 == 0
            or step == len(future_times)
        ):

            print(
                f"Forecasted "
                f"{step:,}/"
                f"{len(future_times):,} "
                f"intervals through "
                f"{timestamp}"
            )


    # ========================================================
    # BUILD FORECAST DATAFRAME
    # ========================================================

    forecast_df = pd.DataFrame(
        forecast_rows
    )


    # ========================================================
    # CREATE OUTPUT DIRECTORIES
    # ========================================================

    Path(
        forecast_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    Path(
        forecast_archive_dir
    ).mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # SAVE CURRENT FORECAST
    # ========================================================

    forecast_df.to_csv(
        forecast_path,
        index=False
    )

    print(
        f"\nSaved "
        f"{len(forecast_df):,} predictions "
        f"to {forecast_path}"
    )


    # ========================================================
    # SAVE ARCHIVE FORECAST
    # ========================================================

    archive_path = (
        f"{forecast_archive_dir}/"
        f"traffic_forecast_"
        f"{last_trained_timestamp.strftime('%Y%m%d')}"
        f".csv"
    )


    forecast_df.to_csv(
        archive_path,
        index=False
    )

    print(
        "Saved archive forecast file:",
        archive_path
    )


    # ========================================================
    # PREVIEW
    # ========================================================

    print()
    print(
        forecast_df.head()
    )

    print()
    print(
        forecast_df.tail()
    )


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        "forecast": forecast_df,
        "forecast_path": forecast_path,
        "archive_path": archive_path,
        "forecast_start": forecast_start,
        "forecast_end": forecast_end,
        "rows": len(forecast_df)
    }


if __name__ == "__main__":
    run_forecast_generation()
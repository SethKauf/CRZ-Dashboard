import json
import shutil

import joblib
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from src.model_functions import (
    make_update_group_dataset,
    format_for_model
)


# ============================================================
# PATHS / CONSTANTS
# ============================================================

DATA_PATH = "data/modeling_data.csv"
MODEL_PATH = "models/lstm_traffic_model.keras"
SCALER_PATH = "models/traffic_scaler.joblib"
CONFIG_PATH = "src/model_config.json"

NUMERIC_FEATURES = [
    "traffic_scaled",
    "holiday_ind",
    "overnight_ind",
    "dow_sin",
    "dow_cos"
]


def run_model_update(
    data_path=DATA_PATH,
    model_path=MODEL_PATH,
    scaler_path=SCALER_PATH,
    config_path=CONFIG_PATH,
    epochs=3,
    learning_rate=0.0001
):
    """
    Fine-tune the existing LSTM on newly appended traffic data.

    Uses the existing scaler without refitting it.

    Returns
    -------
    dict
        Metadata about the update run.
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
    batch_size = config["batch_size"]

    new_data_start = (
        last_trained_timestamp
        + pd.Timedelta(minutes=10)
    )

    print(
        "\nModel last trained through:",
        last_trained_timestamp
    )

    print(
        "New training data starts:",
        new_data_start
    )


    # ========================================================
    # LOAD DATA
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


    # ========================================================
    # LATEST COMMON TIMESTAMP
    # ========================================================

    latest_common_timestamp = (
        data
        .groupby("group_id")[
            "toll_10_minute_block"
        ]
        .max()
        .min()
    )

    print(
        "Latest common data timestamp:",
        latest_common_timestamp
    )


    # ========================================================
    # NOTHING NEW? RETURN CLEANLY
    # ========================================================

    if latest_common_timestamp < new_data_start:

        print(
            "No new data available. "
            "Model update not required."
        )

        return {
            "updated": False,
            "last_trained_timestamp":
                last_trained_timestamp,
            "latest_common_timestamp":
                latest_common_timestamp
        }


    # Only train through timestamp shared by all groups
    data = data[
        data["toll_10_minute_block"]
        <= latest_common_timestamp
    ].copy()


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
    # ARCHIVE CURRENT MODEL
    # ========================================================

    archive_date = (
        last_trained_timestamp
        .strftime("%Y%m%d")
    )

    model_archive_ref = (
        f"models/"
        f"trained_model_{archive_date}.keras"
    )

    shutil.copy2(
        model_path,
        model_archive_ref
    )

    print(
        "Archived existing model to:",
        model_archive_ref
    )


    # ========================================================
    # APPLY EXISTING SCALER
    # ========================================================

    data["traffic_scaled"] = (
        traffic_scaler
        .transform(
            data[["traffic_volume"]]
        )
        .ravel()
    )


    # ========================================================
    # BUILD UPDATE DATASET
    # ========================================================

    update_ds = None

    for group_id, group_df in data.groupby(
        "group_id"
    ):

        group_ds = make_update_group_dataset(
            group_df=group_df,
            numeric_features=NUMERIC_FEATURES,
            new_data_start=new_data_start,
            seq_length=seq_length
        )

        if group_ds is None:
            continue

        if update_ds is None:
            update_ds = group_ds

        else:
            update_ds = (
                update_ds
                .concatenate(group_ds)
            )


    if update_ds is None:

        print(
            "No new training sequences found."
        )

        return {
            "updated": False,
            "last_trained_timestamp":
                last_trained_timestamp,
            "latest_common_timestamp":
                latest_common_timestamp
        }


    # ========================================================
    # BATCH DATASET
    # ========================================================

    update_ds = (
        update_ds
        .shuffle(20_000)
        .batch(batch_size)
        .map(
            format_for_model,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .prefetch(
            tf.data.AUTOTUNE
        )
    )


    # ========================================================
    # COMPILE FOR FINE-TUNING
    # ========================================================

    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate
        ),
        loss="mse",
        metrics=["mae"]
    )


    # ========================================================
    # UPDATE MODEL
    # ========================================================

    history = model.fit(
        update_ds,
        epochs=epochs
    )


    # ========================================================
    # SAVE UPDATED MODEL
    # ========================================================

    model.save(
        model_path
    )

    print(
        "Updated model saved to:",
        model_path
    )


    # ========================================================
    # UPDATE CONFIG
    # ========================================================

    config["last_trained_timestamp"] = (
        latest_common_timestamp
        .isoformat()
    )

    with open(config_path, "w") as f:

        json.dump(
            config,
            f,
            indent=4
        )

    print(
        "Model now trained through:",
        latest_common_timestamp
    )


    # ========================================================
    # RETURN UPDATE INFO
    # ========================================================

    return {
        "updated": True,
        "history": history.history,
        "previous_trained_timestamp":
            last_trained_timestamp,
        "last_trained_timestamp":
            latest_common_timestamp,
        "archive_path":
            model_archive_ref
    }


if __name__ == "__main__":
    run_model_update()
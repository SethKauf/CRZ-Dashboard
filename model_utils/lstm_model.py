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
# PATHS
# ============================================================

DATA_PATH = "data/modeling_data.csv"
MODEL_PATH = "models/lstm_traffic_model.keras"
SCALER_PATH = "models/traffic_scaler.joblib"
CONFIG_PATH = "src/model_config.json"


# ============================================================
# MODEL FEATURES
# ============================================================

numeric_features = [
    "traffic_scaled",
    "holiday_ind",
    "overnight_ind",
    "dow_sin",
    "dow_cos"
]


# ============================================================
# LOAD MODEL CONFIG
# ============================================================

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)


LAST_TRAINED_TIMESTAMP = pd.Timestamp(
    config["last_trained_timestamp"]
)

SEQ_LENGTH = config["seq_length"]
BATCH_SIZE = config["batch_size"]


new_data_start = (
    LAST_TRAINED_TIMESTAMP
    + pd.Timedelta(minutes=10)
)


print(
    "\nModel last trained through:",
    LAST_TRAINED_TIMESTAMP
)

print(
    "\nNew training data starts:",
    new_data_start
)


# ============================================================
# LOAD DATA
# ============================================================

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
    .reset_index(drop=True)
)


# ============================================================
# DETERMINE LATEST COMMON TIMESTAMP
# ============================================================

latest_common_timestamp = (
    data
    .groupby("group_id")[
        "toll_10_minute_block"
    ]
    .max()
    .min()
)


print(
    "\nLatest common data timestamp:",
    latest_common_timestamp
)


# Exit cleanly if there is nothing new
if latest_common_timestamp < new_data_start:

    print(
        "\nNo new data available. "
        "\nModel update not required."
    )

    raise SystemExit


# Only train through the latest timestamp
# shared by every group.
data = data[
    data["toll_10_minute_block"]
    <= latest_common_timestamp
].copy()


# ============================================================
# LOAD MODEL + SCALER
# ============================================================

model = load_model(
    MODEL_PATH,
    compile=False
)

traffic_scaler = joblib.load(
    SCALER_PATH
)


# ============================================================
# ARCHIVE CURRENT MODEL
# ============================================================

archive_date = (
    LAST_TRAINED_TIMESTAMP
    .strftime("%Y%m%d")
)

model_archive_ref = (
    f"models/"
    f"trained_model_{archive_date}.keras"
)

shutil.copy2(
    MODEL_PATH,
    model_archive_ref
)

print(
    "\nArchived existing model to:",
    model_archive_ref
)


# ============================================================
# APPLY EXISTING SCALER
# ============================================================

# IMPORTANT:
# transform only -- never refit the scaler here.

data["traffic_scaled"] = (
    traffic_scaler
    .transform(
        data[["traffic_volume"]]
    )
    .ravel()
)


# ============================================================
# BUILD UPDATE DATASET
# ============================================================

update_ds = None


for group_id, group_df in data.groupby(
    "group_id"
):

    group_ds = make_update_group_dataset(
        group_df=group_df,
        numeric_features=numeric_features,
        new_data_start=new_data_start,
        seq_length=SEQ_LENGTH
    )

    # Group has no genuinely new observations
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
    raise SystemExit


# ============================================================
# BATCH DATASET
# ============================================================

update_ds = (
    update_ds
    .shuffle(20_000)
    .batch(BATCH_SIZE)
    .map(
        format_for_model,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .prefetch(
        tf.data.AUTOTUNE
    )
)


# ============================================================
# COMPILE FOR FINE-TUNING
# ============================================================

model.compile(
    optimizer=Adam(
        learning_rate=0.0001
    ),
    loss="mse",
    metrics=["mae"]
)


# ============================================================
# UPDATE MODEL
# ============================================================

history = model.fit(
    update_ds,
    epochs=3
)


# ============================================================
# SAVE UPDATED MODEL
# ============================================================

model.save(
    MODEL_PATH
)

print(
    "Updated model saved to:",
    MODEL_PATH
)


# ============================================================
# UPDATE MODEL CONFIG
# ============================================================

# Only happens AFTER model.save() succeeds.

config["last_trained_timestamp"] = (
    latest_common_timestamp
    .isoformat()
)


with open(CONFIG_PATH, "w") as f:

    json.dump(
        config,
        f,
        indent=4
    )


print(
    "Model now trained through:",
    latest_common_timestamp
)
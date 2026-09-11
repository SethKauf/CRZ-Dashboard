import pandas as pd
import numpy as np
import tensorflow as tf


def make_update_group_dataset(
    group_df,
    numeric_features,
    new_data_start,
    seq_length=144
):
    """
    Create LSTM training sequences where:
      - the previous seq_length rows may come from old data
      - every target is from new_data_start onward
    """

    group_df = (
        group_df
        .sort_values("toll_10_minute_block")
        .reset_index(drop=True)
    )

    # Historical context only
    history = (
        group_df[
            group_df["toll_10_minute_block"] < new_data_start
        ]
        .tail(seq_length)
    )

    # Rows the current model has not trained on
    new_data = group_df[
        group_df["toll_10_minute_block"] >= new_data_start
    ]

    if len(history) < seq_length:
        raise ValueError(
            f"Need {seq_length} historical rows, "
            f"but only found {len(history)}."
        )

    if new_data.empty:
        return None

    # Exactly 144 old rows + all new rows
    update_data = pd.concat(
        [history, new_data],
        ignore_index=True
    )

    numeric = update_data[
        numeric_features
    ].to_numpy(dtype=np.float32)

    region = update_data[
        "region_id"
    ].to_numpy(dtype=np.int32)

    group = update_data[
        "group_id"
    ].to_numpy(dtype=np.int32)

    target = update_data[
        "traffic_scaled"
    ].to_numpy(dtype=np.float32)


    numeric_ds = tf.keras.utils.timeseries_dataset_from_array(
        data=numeric[:-1],
        targets=None,
        sequence_length=seq_length,
        sequence_stride=1,
        shuffle=False,
        batch_size=None
    )

    region_ds = tf.keras.utils.timeseries_dataset_from_array(
        data=region[:-1],
        targets=None,
        sequence_length=seq_length,
        sequence_stride=1,
        shuffle=False,
        batch_size=None
    )

    group_ds = tf.keras.utils.timeseries_dataset_from_array(
        data=group[:-1],
        targets=None,
        sequence_length=seq_length,
        sequence_stride=1,
        shuffle=False,
        batch_size=None
    )

    # Since update_data begins with exactly seq_length historical
    # rows, every target here is genuinely new data.
    target_ds = tf.data.Dataset.from_tensor_slices(
        target[seq_length:]
    )

    return tf.data.Dataset.zip(
        (
            numeric_ds,
            region_ds,
            group_ds,
            target_ds
        )
    )


def format_for_model(
    numeric,
    region,
    group,
    target
):
    return {
        "numeric": numeric,
        "region": region,
        "group": group
    }, target
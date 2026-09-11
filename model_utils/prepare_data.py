import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.fetch_data import fetch_traffic_data, process_data

CURRENT_DATA_PATH = "data/modeling_data.csv"

def run_prepare_data(
    current_data_path=CURRENT_DATA_PATH,
    full_pull=False
):
    """
    Fetch new CRZ data, process it to modeling format,
    merge it into the existing modeling dataset,
    and save the updated file.

    Returns
    -------
    pd.DataFrame
        Updated modeling dataset.
    """

    # --------------------------------------------------------
    # Load current modeling data
    # --------------------------------------------------------

    current_data = pd.read_csv(
        current_data_path,
        index_col=None
    )

    current_data["toll_10_minute_block"] = pd.to_datetime(
        current_data["toll_10_minute_block"]
    )


    # --------------------------------------------------------
    # Pull new API data
    # --------------------------------------------------------

    new_data = fetch_traffic_data(
        current_data_path,
        full_pull=full_pull
    )


    if new_data.empty:
        print("No new API data returned.")
        return current_data


    # --------------------------------------------------------
    # Archive raw API pull
    # --------------------------------------------------------

    todays_date = datetime.today().strftime("%Y%m%d")

    api_pull_path = (
        f"data/api_pull_data/"
        f"crz_data_pull_{todays_date}.csv"
    )

    new_data.to_csv(
        api_pull_path,
        index=False
    )

    print(
        f"Saved raw API pull to {api_pull_path}"
    )


    # --------------------------------------------------------
    # Process incoming data into modeling format
    # --------------------------------------------------------

    append_data = process_data(
        new_data
    )


    # --------------------------------------------------------
    # Identify columns defining a unique modeling observation
    # --------------------------------------------------------

    key_cols = append_data.columns[
        append_data.columns.get_loc(
            "toll_10_minute_block"
        ):
        append_data.columns.get_loc(
            "traffic_volume"
        ) + 1
    ].tolist()


    # --------------------------------------------------------
    # Merge
    #
    # Put current data FIRST and append data SECOND so
    # keep="last" preserves the newly pulled version.
    # --------------------------------------------------------

    data = (
        pd.concat(
            [
                current_data,
                append_data
            ],
            ignore_index=True
        )
        .drop_duplicates(
            subset=key_cols,
            keep="last"
        )
        .sort_values(
            [
                "group_id",
                "toll_10_minute_block"
            ]
        )
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # Save updated modeling dataset
    # --------------------------------------------------------

    data.to_csv(
        current_data_path,
        index=False
    )


    print(
        "\n"
        "Saved updated modeling data.\n"
        f"Min timestamp: "
        f"{data['toll_10_minute_block'].min()}\n"
        f"Max timestamp: "
        f"{data['toll_10_minute_block'].max()}\n"
    )


    return data


if __name__ == "__main__":
    run_prepare_data()
import requests
import pandas as pd
import numpy as np
import holidays
from datetime import datetime, timedelta

from src.mappings import REGION_MAPPING, GROUP_MAPPING

def fetch_traffic_data(DATA_PATH,full_pull=False):
    """
    Fetch CRZ traffic data from NY Gov API
    Data is pulled -3 days and on from current max date in data.
    """
    # get range start date
    current_data = pd.read_csv(
        DATA_PATH
    )

    current_data['toll_10_minute_block'] = pd.to_datetime(
        current_data['toll_10_minute_block']
    )

    current_max_date = (
        current_data['toll_10_minute_block'].max()
    )

    api_start_date = (
        current_max_date - pd.Timedelta(days=3)
    )

    print(
        f"Current data max date: {current_max_date}."
    )

    if full_pull == False:
            
        print(
            f"API will pull data beginning on: {api_start_date}.\n"
        )

    else:

        print("Initializing full data pull.\n")

    # api setup

    url = "https://data.ny.gov/resource/t6yz-b64h.json"

    chunk_size = 50_000
    offset = 0

    chunks = []

    # format date for Socrata API

    api_date_string = api_start_date.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )

    # fetch new data

    while True:

        print(

            f"Fetching rows "
            f"{offset + 1:,}-"
            f"{offset + chunk_size:,} ...\n"

        )

        if full_pull == False:

            params = {
                "$limit": chunk_size,
                "$offset": offset,

                "$where": (
                    f"toll_10_minute_block > '{api_date_string}'"
                ),

                "$order":"toll_10_minute_block ASC"
            }

        else:

            params = {
                "$limit": chunk_size,
                "$offset": offset,

                "$order":"toll_10_minute_block ASC"
            }

        response = requests.get(
            url,
            params=params,
            timeout=60
        )

        # raise error if unsuccesful response
        response.raise_for_status()
        
        api_data = response.json()

        if len(api_data) == 0:

            print("No more data to fetch.\n")

            break

        chunks.append(
            pd.DataFrame(api_data)
        )

        offset += chunk_size
    
    # combine chunks
    
    if not chunks:

        print("No new API data found.\n")
        return pd.DataFrame()

    new_data = pd.concat(
        chunks,
        ignore_index=True
    )

    # convert timestamp immediately
    new_data['toll_10_minute_block'] = pd.to_datetime(new_data['toll_10_minute_block'])

    print(
        f"Fetched {len(new_data):,} rows."
    )

    print(
        "API date range:",
        new_data["toll_10_minute_block"].min(),
        "to",
        new_data["toll_10_minute_block"].max()
    )

    return new_data

# format new data to match old
def process_data(new_data):
    """
    Processes newly pulled API data to match formatting of current processed data.
    """
    # holiday and overnight inds
    new_data['holiday_ind'] = new_data['toll_date'].apply(lambda x: 1 if x in holidays.US() else 0)
    new_data['overnight_ind'] = new_data['time_period'].apply(lambda x: 1 if x == 'Overnight' else 0)

    # group and region ids
    new_data['region_id'] = (
        new_data['detection_region']
        .map(REGION_MAPPING)
        .fillna(0)
        .astype(int)
    )

    new_data['group_id'] = (
        new_data['detection_group']
        .map(GROUP_MAPPING)
        .fillna(0)
        .astype(int)
    )

    # day of week sin/cos
    new_data['day_of_week_int'] = pd.to_numeric(new_data['day_of_week_int'])

    new_data['dow_sin'] = np.sin(
        2 * np.pi * new_data['day_of_week_int'] / 7
    )

    new_data['dow_cos'] = np.cos(
        2 * np.pi * new_data['day_of_week_int'] / 7
    )

    # group relevant cols, summing crz_entries + excluded_roadway_entries
    keep_cols = ['toll_10_minute_block','day_of_week_int','holiday_ind','overnight_ind','region_id','group_id','dow_sin','dow_cos']

    new_data['crz_entries'] = pd.to_numeric(new_data['crz_entries'])

    new_data['excluded_roadway_entries'] = pd.to_numeric(new_data['excluded_roadway_entries'])

    grouped_data = (
        new_data
        .groupby(keep_cols,as_index=False)
        .agg(
            crz_entries=('crz_entries','sum'),
            excluded_roadway_entries=('excluded_roadway_entries','sum')
        )
    )

    grouped_data['traffic_volume'] = (
        grouped_data['crz_entries']
        + grouped_data['excluded_roadway_entries']
    )

    grouped_data.drop(['crz_entries','excluded_roadway_entries'],axis=1,inplace=True)

    return grouped_data
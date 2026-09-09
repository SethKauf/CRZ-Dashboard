import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.fetch_data import fetch_traffic_data, process_data

CURRENT_DATA_PATH = "data/modeling_data.csv"

current_data = pd.read_csv(CURRENT_DATA_PATH,index_col=None)

# convert timestamp to datetime
current_data['toll_10_minute_block'] = pd.to_datetime(current_data['toll_10_minute_block'])

# pull new data
new_data = fetch_traffic_data(CURRENT_DATA_PATH,full_pull=False)

# save data append pull
todays_date = datetime.today().strftime("%Y%m%d")

new_data.to_csv(f"data/api_pull_data/crz_data_pull_{todays_date}.csv",index=False)

# process data to match previous format
append_data = process_data(new_data)

# All columns from toll_date through detection_region
key_cols = append_data.columns[
    append_data.columns.get_loc("toll_10_minute_block"):
    append_data.columns.get_loc("traffic_volume") + 1
].tolist()

# append data supercedes current
data = (
    pd.concat([append_data, current_data], ignore_index=True)
      .drop_duplicates(subset=key_cols, keep="last")
      .reset_index(drop=True)
)

data.to_csv(CURRENT_DATA_PATH,index=False)

print(f"""
Saved new modeling data with min timestamp of {data['toll_10_minute_block'].min()} and max timestamp of {data['toll_10_minute_block'].max()}.
""")
import streamlit as st
import folium
from streamlit_folium import st_folium

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import date, timedelta

from tensorflow.keras.models import load_model

# ============================================================ #
########################### CONFIG ###########################
# ============================================================ #
st.set_page_config(
    page_title="NYC Congestion Relief Zone Forecast",
    layout="wide"
)

MODEL_PATH = "models/lstm_traffic_model.keras"
SCALER_PATH = "models/traffic_scaler.joblib"

DATA_PATH = "data/traffic_data.parquet"

SEQ_LENGTH = 144

numeric_features = [
    "traffic_scaled",
    "holiday_ind",
    "overnight_ind",
    "dow_sin",
    "dow_cos"
]

# ============================================================ #
##################### REGION/GROUP MAPPING #####################    
# ============================================================ #
REGION_MAPPING = {
    "Brooklyn": 1,
    "Queens": 2,
    "New Jersey": 3,
    "West Side Highway": 4,
    "FDR Drive": 5,
    "West 60th St": 6,
    "East 60th St": 7,
}

GROUP_MAPPING = {
    "Brooklyn Bridge": 1,
    "Hugh L. Carey Tunnel": 2,
    "Williamsburg Bridge": 3,
    "Manhattan Bridge": 4,
    "Queensboro Bridge": 5,
    "Queens Midtown Tunnel": 6,
    "Holland Tunnel": 7,
    "Lincoln Tunnel": 8,
    "West Side Highway at 60th St": 9,
    "FDR Drive at 60th St": 10,
    "West 60th St": 11,
    "East 60th St": 12,
}

# group_id -> region_id
GROUP_TO_REGION = {
    1:1, # Brooklyn Bridge
    2:1, # Hugh L. Carey Tunnel
    3:1, # Williamsburg Bridge
    4:1, # Manhattan Bridge
    5:2, # Queensboro Bridge
    6:2, # Queens Midtown Tunnel
    7:3, # Holland Tunnel
    8:3, # Lincoln Tunnel
    9:4, # West Side Highway at 60th St
    10:5, # FDR Drive at 60th St
    11:6, # West 60th St
    12:7, # East 60th St
}

# Approximate coordinates for visualization.
GROUP_COORDS = {
    1: [40.7061, -73.9969],
    2: [40.7027, -74.0151],
    3: [40.7132, -73.9727],
    4: [40.7075, -73.9904],
    5: [40.7576, -73.9566],
    6: [40.7465, -73.9715],
    7: [40.7273, -74.0117],
    8: [40.7604, -74.0027],
    9: [40.7715, -73.9905],
    10: [40.7600, -73.9580],
    11: [40.7700, -73.9870],
    12: [40.7620, -73.9660],
}

GROUP_NAME_BY_ID = {
    group_id: name
    for name, group_id in GROUP_MAPPING.items()
}

REGION_NAME_BY_ID = {
    region_id: name
    for name, region_id in REGION_MAPPING.items()
}

# ============================================================ #
##################### LOAD MODEL + SCALER ######################    
# ============================================================ #

@st.cache_resource
def load_model_and_scaler():

    model = load_model(
        MODEL_PATH,
        compile=False
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    return model, scaler


model, traffic_scaler = load_model_and_scaler()

# ============================================================ #
###################### LOAD WEEKLY DATA ########################
# ============================================================ #

@st.cache_data
def load_traffic_data():
    path = Path(DATA_PATH)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    df["toll_10_minute_block"] = pd.to_datetime(
        df["toll_10_minute_block"]
    )

    return df

data = load_traffic_data()

# ============================================================ #
###################### FEATURE HELPERS #########################
# ============================================================ #
# overnight ind
def get_overnight_ind(data):
    """
    Create overnight indicator based on time_period
    """
    data['overnight_ind'] = data['time_period'].apply(
        lambda x: 1 if x == 'Overnight' else 0
    ).astype(int)

    return data

# add holidays
def get_holiday_ind(data):
    """
    Create holiday indicator based on toll_date
    """

    us_holidays = holidays.US()

    data['holiday_ind'] = data['toll_date'].apply(
        lambda x: x in us_holidays
    ).astype(int)

    return data

def create_future_features(timestamp):
    # dataset comes with day of week as int
    day_of_week_int = data['day_of_week_int']

    return {
        "toll_10_minute_block": timestamp,
        "holiday_ind":get_holiday_ind(data),
        "overnight_ind":get_overnight_ind(data),
        "dow_sin": np.sin(2 * np.pi * day_of_week_int / 7),
        "dow_cos": np.cos(2 * np.pi * day_of_week_int / 7),
    }

# ============================================================ #
################## PREPARE HISTORICAL DATA #####################
# ============================================================ #

def prepare_group_history(
        df,
        group_id,
        scaler,
        latest_timestamp
):
    group_df = (
        df[
            (df['group_id'] == group_id)
            &
            (df['toll_10_minute_block'] <= latest_timestamp)
        ]
        .sort_values("toll_10_minute_block")
        .tail(SEQ_LENGTH)
        .copy()
    )

    if len(group_df) < SEQ_LENGTH:
        raise ValueError(
            f"Group {group_id} only has {len(group_df)} records, but SEQ_LENGTH is {SEQ_LENGTH}."
        )

    group_df['traffic_scaled'] = scaler.transform(
        group_df[['traffic_volume']]
    )

    # recreating day of week sin/cos incase they don't already exist
    group_df['dow_sin'] = np.sin(2 * np.pi * group_df['day_of_week_int'] / 7)
    group_df['dow_cos'] = np.cos(2 * np.pi * group_df['day_of_week_int'] / 7)

    return group_df[
        [
            "toll_10_minute_block",
            *NUMERIC_FEATURES
        ]
    ].copy()

# ============================================================ #
#################### RECURSIVE FORECAST ########################
# ============================================================ #

@st.cache_data
def generate_forecast(
    forecast_date
):
    # all groups should forecast from the same starting point
    latest_by_group = (
        data.groupby("group_id")[
            "toll_10_minute_block"
        ]
        .max()
    )

    latest_timestamp = latest_by_group.min()

    histories = {}

    for group_id in GROUP_NAME_BY_ID:

        histories[group_id] = prepare_group_history(
            data,
            group_id,
            traffic_scaler,
            latest_timestamp
        )

    target_start = pd.Timestamp(forecast_date)

    target_end = (
        target_start
        + pd.Timedelta(days=1)
        - pd.Timedelta(minutes=10)
    )

    if target_end <= latest_timestamp:
        raise ValueError(
            """Select forecast date is not after the latest available observation."""
        )

    future_times = pd.date_range(
        start=latest_timemstamp + pd.Timedelta(minutes=10),
        end=target_end,
        freq="10min"
    )

    forecast_rows = []

    for timestamp in future_times:

        numeric_batch = []
        region_batch = []
        group_batch = []

        group_ids = sorted(GROUP_NAME_BY_ID)

        # ----------------------------------------
        # build one batch containing all 12 groups
        # ----------------------------------------

        for group_id in group_ids:
            history = histories[group_id].tail(
                SEQ_LENGTH
            )

            numeric_batch.append(
                history[
                    NUMERIC_FEATURES
                ].to_numpy(
                    dtype=np.float32
                )
            )

            region_id = GROUP_TO_REGION[group_id]

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

        group_stack = np.stack(
            group_batch
        )

        # -----------------------------
        # predict all 12 groups at once
        # -----------------------------

        predictions_scaled = model.predict(
            {
                "numeric":numeric_batch,
                "region":region_batch,
                "group":group_batch
            },
            verbose=0
        ).reshape(-1)

        predictions_actual = (
            traffic_scaler
            .inverse_transform(
            predictions_scaled.reshape(-1,1)
            )
            .reshape(-1)
        )

        # -------------------------------
        # append predictions into history
        # -------------------------------

        future_features = create_future_features(
            timestamp
        )

        for i, group_id in enumerate(group_ids):

            new_row = {
                **future_features,
                "traffic_scaled":
                    predictions_scaled[i],
            }

            histories[group_id] = pd.concat(
                [
                    histories[group_id],
                    pd.DataFrame([new_row])
                ],
                ignore_index=True
            )

            # only save rows for requested date
            if timestamp >= target_start:

                forecast_rows.append(
                    {
                        "toll_10_minute_block":
                            timestamp,

                        "group_id":
                            group_id,

                        "region_id":
                            GROUP_TO_REGION[
                                group_id
                            ],

                        "group_name":
                            GROUP_NAME_BY_ID[
                                group_id
                            ],

                        "predicted_traffic":
                            predictions_actual[i],
                    }
                )

    return pd.DataFrame(
        forecast_rows
    )

# ============================================================ #
######################## PAGE TITLE ############################
# ============================================================ #

st.title(
    "NYC Congestion Relief Zone Entryways Traffic Forecast"
)

st.caption(
    "LSTM-based speculative 10-minute traffic forecasts"
)

# ============================================================ #
######################### SIDEBAR ##############################
# ============================================================ #

st.sidebar.header(
    "Forecast Controls"
)

forecast_date = st.sidebar.date_input(
    "Forecast Date",
    value=date.today() + timedelta(days=1)
)

region_selection = st.sidebar.selectbox(
    "Detection Region",
    [
        "All Regions",
        *REGION_MAPPING.keys()
    ]
)

map_style = st.sidebar.selectbox(
    "Map Style",
    [
        "Light Mode",
        "Dark Mode",
        "OpenStreetMap"
    ]
)

generate = st.sidebar.button(
    "Generate Forecast"
)

# ============================================================ #
#################### GENERATE FORECAST #########################
# ============================================================ #

if generate:

    with st.spinner(
        f"Forecasting traffic for {forecast_date}..."
    ):
        forecast = generate_forecast(
            forecast_date
        )

    st.session_state['forecast'] = forecast

forecast = st.session_state.get(
    "forecast"
)

# ============================================================ #
############################ MAP ###############################
# ============================================================ #

tiles_dict = {
    "OpenStreetMap":
        "OpenStreetMap",

    "Light Mode":
        "CartoDB Positron",

    "Dark mode":
        "CartoDB Dark_Matter",
}

m = folium.Map(
    location = [40.735, -74.000],
    zoom_start=12,
    tiles=tiles_dict[map_style],
    control_scale=True
)

# ============================================================ #
################### ADD ALL 12 LOCATIONS #######################
# ============================================================ #

for group_id, group_name in GROUP_NAME_BY_ID.items():

    region_id = GROUP_TO_REGION[
        group_id
    ]

    region_name = REGION_NAME_BY_ID[
        region_id
    ]

    # apply optional region filter
    if (
        region_selection != "All Regions"
        and region_name != region_selection
    ):
        continue

    coords = GROUP_COORDS[
        group_id
    ]

    # -------------------
    # forecast statistics
    # -------------------

    if forecast is not None:

        group_forecast = forecast[
            forecast["group_id"] == group_id
        ]

        if len(group_forecast) > 0:

            daily_mean = (
                group_forecast[
                    "predicted_traffic"
                ]
                .mean()
            )

            daily_peak = (
                group_forecast[
                    "predicted_traffic"
                ]
                .max()
            )

            peak_row = (
                group_forecast.loc[
                    group_forecast[
                        "predicted_traffic"
                    ].idxmax()
                ]
            )

            peak_time = (
                peak_row[
                    "toll_10_minute_block"
                ]
                .strftime("%I:%M %p")
            )

            popup_html = f"""
            <b>{group_name}</b><br>
            {region_name}<br><br>

            <b>Forecast date:</b>
            {forecast_date}<br>

            <b>Average 10-min volume:</b>
            {daily_mean:.0f}<br>

            <b>Peak 10-min volume:</b>
            {daily_peak:.0f}<br>

            <b>Peak time:</b>
            {peak_time}
            """

            tooltip = (
                f"{group_name}: "
                f"Peak {daily_peak:.0f}"
            )

        else:

            popup_html = f"""
            <b>{group_name}</b><br>
            No forecast available
            """

            tooltip = group_name

    else:

        popup_html = f"""
        <b>{group_name}</b><br>
        {region_name}<br><br>
        Generate a forecast to view traffic.
        """

        tooltip = group_name

    folium.Marker(
        location = coords,

        popup=folium.Popup(
            popup_html,
            max_width=300
        ),

        tooltip=tooltip,

        icon=folium.Icon(
            color="blue",
            icon="info-sign"
        )
    ).add_to(m)


# Expanded bounds to include 60th Street Locations and Lower Manhattan crossings

m.fit_bounds(
    [
        [40.695, -74.025],
        [40.780, -73.945]
    ]
)

st_folium(
    m,
    width=1200,
    height=600,
    returned_objects=[]
)

# ============================================================ #
##################### FORECAST CHART ###########################
# ============================================================ #

if forecast is not None:

    st.subheader(
        f"Traffic Forecast - {forecast_date}"
    )

    selected_group_name = st.selectbox(
        "View forecast curve",
        list(GROUP_MAPPING.keys())
    )

    selected_group_id = GROUP_MAPPING[
        selected_group_name
    ]

    plot_data = forecast[
        forecast["group_id"] == selected_group_id
    ].copy()

    fig, ax = plt.subplots(
        figsize=(12,5)
    )

    ax.plot(
        plot_data[
            "toll_10_minute_block"
        ],

        plot_data[
            "predicted_traffic"
        ],

        linewidth=2
    )

    ax.set_title(
        f"{selected_group_name} "
        f"- {forecast_date}"
    )

    ax.set_xlabel(
        "Time"
    )

    ax.set_ylabel(
        "Predicted Traffic Volume"
    )

    ax.grid(True)

    fig.autofmt_xdate()

    st.pyplot(
        fig
    )

# ============================================================ #
########################## ABOUT ###############################
# ============================================================ #

st.sidebar.markdown("---")

st.sidebar.subheader(
    "About"
)

st.sidebar.write(
    """
    Forecasts are generated using a global LSTM, trained across all 12 NYC CBD traffic groups.

    Each prediction represents estimated traffic volume during a 10-minute interval.
    """
)
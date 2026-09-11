import streamlit as st
import folium
from streamlit_folium import st_folium

import pandas as pd

from src.mappings import (
    REGION_MAPPING,
    GROUP_MAPPING,
    GROUP_TO_REGION
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Lower Manhattan Traffic Forecast",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

FORECAST_PATH = "data/forecasts/traffic_forecast.csv"


# ============================================================
# GROUP / REGION LOOKUPS
# ============================================================

GROUP_NAME_BY_ID = {
    group_id: group_name
    for group_name, group_id in GROUP_MAPPING.items()
}

REGION_NAME_BY_ID = {
    region_id: region_name
    for region_name, region_id in REGION_MAPPING.items()
}


# ============================================================
# GROUP COORDINATES
# ============================================================

# Approximate map coordinates.
# Replace with your preferred/canonical coordinates later if needed.

GROUP_COORDS = {
    1: [40.7061, -73.9969],   # Brooklyn Bridge
    2: [40.7027, -74.0151],   # Hugh L. Carey Tunnel
    3: [40.7132, -73.9727],   # Williamsburg Bridge
    4: [40.7075, -73.9904],   # Manhattan Bridge

    5: [40.7576, -73.9566],   # Queensboro Bridge
    6: [40.7465, -73.9715],   # Queens Midtown Tunnel

    7: [40.7273, -74.0117],   # Holland Tunnel
    8: [40.7604, -74.0027],   # Lincoln Tunnel

    9: [40.7715, -73.9905],   # West Side Highway at 60th St
    10: [40.7600, -73.9580],  # FDR Drive at 60th St
    11: [40.7700, -73.9870],  # West 60th St
    12: [40.7620, -73.9660],  # East 60th St
}


# ============================================================
# LOAD FORECAST DATA
# ============================================================

@st.cache_data
def load_forecast_data():

    df = pd.read_csv(
        FORECAST_PATH
    )

    df["toll_10_minute_block"] = pd.to_datetime(
        df["toll_10_minute_block"]
    )

    df["forecast_date"] = (
        df["toll_10_minute_block"].dt.date
    )

    df["forecast_time"] = (
        df["toll_10_minute_block"].dt.strftime("%H:%M")
    )

    return df


forecast = load_forecast_data()


# ============================================================
# TITLE
# ============================================================

st.title(
    "Lower Manhattan Traffic Forecast"
)

st.caption(
    "Predicted traffic volume at NYC CBD entry points "
    "in 10-minute intervals."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Forecast Controls"
)


# --------------------------
# Map style
# --------------------------

map_style = st.sidebar.selectbox(
    "Select Map Style",
    [
        "Light Mode",
        "Dark Mode",
        "OpenStreetMap"
    ]
)


# --------------------------
# Forecast date
# --------------------------

available_dates = sorted(
    forecast["forecast_date"].unique()
)

selected_date = st.sidebar.selectbox(
    "Forecast Date",
    available_dates
)


# --------------------------
# Forecast time
# --------------------------

available_times = sorted(
    forecast.loc[
        forecast["forecast_date"] == selected_date,
        "forecast_time"
    ]
    .unique()
)

selected_time = st.sidebar.selectbox(
    "Forecast Time",
    available_times,
    index=0
)


# --------------------------
# Region
# --------------------------

region_options = [
    "All Regions",
    *REGION_MAPPING.keys()
]

selected_region = st.sidebar.selectbox(
    "Detection Region",
    region_options
)


# --------------------------
# Mobile map positioning
# --------------------------

is_mobile = st.sidebar.checkbox(
    "Mobile View",
    value=False,
    help="Adjust map position for smaller screens."
)


# ============================================================
# FILTER CURRENT FORECAST
# ============================================================

selected_timestamp = pd.Timestamp(
    f"{selected_date} {selected_time}"
)

current_forecast = forecast[
    forecast["toll_10_minute_block"]
    == selected_timestamp
].copy()


# Add human-readable names
current_forecast["group_name"] = (
    current_forecast["group_id"]
    .map(GROUP_NAME_BY_ID)
)

current_forecast["region_name"] = (
    current_forecast["region_id"]
    .map(REGION_NAME_BY_ID)
)


# Apply optional region filter
if selected_region != "All Regions":

    current_forecast = current_forecast[
        current_forecast["region_name"]
        == selected_region
    ]


# ============================================================
# SUMMARY
# ============================================================

st.subheader(
    selected_timestamp.strftime(
        "%A, %B %d, %Y at %I:%M %p"
    )
)


if not current_forecast.empty:

    total_volume = (
        current_forecast["predicted_volume"]
        .sum()
    )

    avg_volume = (
        current_forecast["predicted_volume"]
        .mean()
    )

    busiest_row = (
        current_forecast.loc[
            current_forecast[
                "predicted_volume"
            ].idxmax()
        ]
    )


    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Predicted Volume",
        f"{total_volume:,.0f}"
    )

    col2.metric(
        "Average per Entry Point",
        f"{avg_volume:,.0f}"
    )

    col3.metric(
        "Highest Volume",
        (
            f"{busiest_row['group_name']} "
            f"({busiest_row['predicted_volume']:,.0f})"
        )
    )


# ============================================================
# MAP CONFIGURATION
# ============================================================

tiles_dict = {
    "OpenStreetMap": "OpenStreetMap",
    "Light Mode": "CartoDB Positron",
    "Dark Mode": "CartoDB Dark_Matter"
}


if is_mobile:

    center_coords = [
        40.7350,
        -73.9850
    ]

else:

    center_coords = [
        40.7350,
        -74.0000
    ]


m = folium.Map(
    location=center_coords,
    zoom_start=12,
    tiles=tiles_dict[map_style],
    prefer_canvas=True,
    control_scale=True
)


# Broader bounds because some entry points extend to 60th St
# and across the Hudson / East River.

m.fit_bounds(
    [
        [40.6950, -74.0250],
        [40.7800, -73.9450]
    ]
)


# ============================================================
# ADD FORECAST LOCATIONS
# ============================================================

for _, row in current_forecast.iterrows():

    group_id = int(
        row["group_id"]
    )

    coords = GROUP_COORDS.get(
        group_id
    )

    if coords is None:
        continue


    group_name = row[
        "group_name"
    ]

    region_name = row[
        "region_name"
    ]

    predicted_volume = row[
        "predicted_volume"
    ]


    popup_html = f"""
        <div style="width: 220px;">
            <h4 style="margin-bottom: 5px;">
                {group_name}
            </h4>

            <b>Region:</b>
            {region_name}
            <br>

            <b>Forecast Time:</b>
            {selected_timestamp.strftime("%I:%M %p")}
            <br>

            <b>Predicted Volume:</b>
            {predicted_volume:,.0f} vehicles
        </div>
    """


    tooltip = (
        f"{group_name}: "
        f"{predicted_volume:,.0f} vehicles"
    )


    folium.Marker(
        location=coords,

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


# ============================================================
# DISPLAY MAP
# ============================================================

st_folium(
    m,
    width=1200,
    height=600,
    returned_objects=[]
)


# ============================================================
# ENTRY POINT TABLE
# ============================================================

st.subheader(
    "Entry Point Forecasts"
)


display_table = (
    current_forecast[
        [
            "group_name",
            "region_name",
            "predicted_volume"
        ]
    ]
    .rename(
        columns={
            "group_name":
                "Entry Point",

            "region_name":
                "Region",

            "predicted_volume":
                "Predicted Volume"
        }
    )
    .sort_values(
        "Predicted Volume",
        ascending=False
    )
)


st.dataframe(
    display_table,
    hide_index=True,
    use_container_width=True
)


# ============================================================
# FULL DAY VIEW FOR ONE ENTRY POINT
# ============================================================

st.subheader(
    "Daily Forecast by Entry Point"
)


group_options = list(
    GROUP_MAPPING.keys()
)


selected_group_name = st.selectbox(
    "Select Entry Point",
    group_options
)


selected_group_id = GROUP_MAPPING[
    selected_group_name
]


daily_group_forecast = (
    forecast[
        (forecast["forecast_date"] == selected_date)
        &
        (forecast["group_id"] == selected_group_id)
    ]
    .sort_values(
        "toll_10_minute_block"
    )
)


if not daily_group_forecast.empty:

    chart_data = (
        daily_group_forecast[
            [
                "toll_10_minute_block",
                "predicted_volume"
            ]
        ]
        .set_index(
            "toll_10_minute_block"
        )
    )

    st.line_chart(
        chart_data
    )


# ============================================================
# ABOUT
# ============================================================

st.sidebar.markdown(
    "---"
)

st.sidebar.subheader(
    "About"
)

st.sidebar.write(
    """
    This dashboard displays LSTM-generated
    traffic forecasts for 12 NYC Congestion
    Relief Zone entry points.

    Forecasts are generated in 10-minute
    intervals from a precomputed backend
    forecast file.
    """
)
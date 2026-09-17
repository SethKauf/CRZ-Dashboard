import streamlit as st
import folium
from streamlit_folium import st_folium

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

import altair as alt

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

# CARTO API key
CARTO_API_KEY = st.secrets["CARTO_API_KEY"]

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
# cache streamlit functions
@st.cache_data(ttl=300)
def load_forecast_data():

    df = pd.read_csv(
        FORECAST_PATH
    )

    df["toll_10_minute_block"] = pd.to_datetime(
        df["toll_10_minute_block"]
    )

    df["predicted_volume_rounded"] = (
        df["predicted_volume_rounded"]
        .astype(int)
    )

    df["forecast_date"] = (
        df["toll_10_minute_block"].dt.date
    )

    df["forecast_time"] = (
        df["toll_10_minute_block"].dt.strftime("%H:%M")
    )

    return df

@st.cache_data(ttl=300)
def load_evaluation_data():

    df = pd.read_csv(
        "data/forecast_evaluations/"
        "forecast_actual_comparison.csv"
    )

    df['toll_10_minute_block'] = pd.to_datetime(
        df['toll_10_minute_block']
    )

    df['forecast_generated_date_display'] = (
        pd.to_datetime(
            df['forecast_generated_date'].astype(str),
            format="%Y%m%d"
        )
        .dt.strftime("%Y-%b-%d")
    )

    return df

@st.cache_data(ttl=300)
def load_group_evaluation():

    df = pd.read_csv(
        "data/forecast_evaluations/"
        "evaluation_by_group.csv"
    )

    df['group_name'] = (
        df['group_id']
        .map(GROUP_NAME_BY_ID)
    )

    df['forecast_generated_date_display'] = (
        pd.to_datetime(
            df['forecast_generated_date'].astype(str),
            format="%Y%m%d"
        )
        .dt.strftime("%Y-%b-%d")
    )

    return df

forecast = load_forecast_data()

evaluation = load_evaluation_data()

group_performance = load_group_evaluation()

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
    ],
    accept_new_options=False
)


# --------------------------
# Forecast date / time
# --------------------------

# Current NYC time
now = datetime.now(
    ZoneInfo("America/New_York")
)

# Next 10-minute interval
minutes_to_add = 10 - (now.minute % 10)

next_interval = (
    now
    .replace(second=0, microsecond=0)
    + timedelta(minutes=minutes_to_add)
)

default_date = next_interval.date()
default_time = next_interval.strftime("%H:%M")


# --------------------------
# Forecast date
# --------------------------

available_dates = sorted(
    forecast["forecast_date"].unique()
)

if default_date in available_dates:
    default_date_index = available_dates.index(
        default_date
    )
else:
    default_date_index = 0

selected_date = st.sidebar.selectbox(
    "Forecast Date",
    available_dates,
    index=default_date_index,
    accept_new_options=False
)


# --------------------------
# Forecast time
# --------------------------

available_times = sorted(
    forecast.loc[
        forecast["forecast_date"] == selected_date,
        "forecast_time"
    ].unique()
)

if (
    selected_date == default_date
    and default_time in available_times
):
    default_time_index = available_times.index(
        default_time
    )
else:
    default_time_index = 0

selected_time = st.sidebar.selectbox(
    "Forecast Time",
    available_times,
    index=default_time_index,
    accept_new_options=False
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
    region_options,
    accept_new_options=False
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

# Dashboard Tabs
forecast_tab, performance_tab = st.tabs(
    [
        "Traffic Forecast",
        "Model Performance"
    ]
)

with forecast_tab:
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
            current_forecast["predicted_volume_rounded"]
            .sum()
        )

        avg_volume = (
            current_forecast["predicted_volume_rounded"]
            .mean()
        )

        busiest_row = (
            current_forecast.loc[
                current_forecast[
                    "predicted_volume_rounded"
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
                f"({busiest_row['predicted_volume_rounded']:,.0f})"
            )
        )


    # ============================================================
    # MAP CONFIGURATION
    # ============================================================

    tiles_dict = {
        "OpenStreetMap": "OpenStreetMap",

        "Light Mode": (
            "https://basemaps.cartocdn.com/"
            "rastertiles/light_all/{z}/{x}/{y}.png"
            f"?key={CARTO_API_KEY}"
        ),

        "Dark Mode": (
            "https://basemaps.cartocdn.com/"
            "rastertiles/dark_all/{z}/{x}/{y}.png"
            f"?key={CARTO_API_KEY}"
        )
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
        attr="© OpenStreetMap contributors © CARTO",
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
            "predicted_volume_rounded"
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
                "predicted_volume_rounded"
            ]
        ]
        .rename(
            columns={
                "group_name":
                    "Entry Point",

                "region_name":
                    "Region",

                "predicted_volume_rounded":
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
        group_options,
        accept_new_options=False
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
# MODEL PERFORMANCE TAB
# ============================================================

with performance_tab:

    st.header(
        "Forecast Performance"
    )

    # --------------------------
    # Select forecast version
    # --------------------------

    available_forecasts = sorted(
        evaluation[
            'forecast_generated_date_display'
        ].unique()
    )

    selected_forecast = st.selectbox(
        "Forecast Date",
        available_forecasts,
        accept_new_options=False
    )

    # Filter evaluation data to selected forecast
    evaluation_filtered = evaluation[
        evaluation[
            'forecast_generated_date_display'
        ] == selected_forecast
    ].copy()

    # Filter group performance to selected forecast
    group_performance_filtered = group_performance[
        group_performance[
            'forecast_generated_date_display'
        ] == selected_forecast
    ].copy()


    # ========================================================
    # HEADLINE PERFORMANCE METRICS
    # ========================================================

    overall_mae = (
        evaluation_filtered[
            'absolute_error'
        ].mean()
    )

    average_bias = (
        evaluation_filtered[
            'error'
        ].mean()
    )

    best_group = (
        group_performance_filtered.loc[
            group_performance_filtered[
                'mae'
            ].idxmin()
        ]
    )

    worst_group = (
        group_performance_filtered.loc[
            group_performance_filtered[
                'mae'
            ].idxmax()
        ]
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Mean Absolute Error",
        f"{overall_mae:.1f} vehicles"
    )

    col2.metric(
        "Average Bias",
        f"{average_bias:+.1f} vehicles"
    )

    col3.metric(
        "Best Predicted",
        GROUP_NAME_BY_ID[
            int(best_group['group_id'])
        ],
        f"MAE {best_group['mae']:.1f}"
    )

    col4.metric(
        "Worst Predicted",
        GROUP_NAME_BY_ID[
            int(worst_group['group_id'])
        ],
        f"MAE {worst_group['mae']:.1f}"
    )


    # ========================================================
    # ACTUAL VS PREDICTED
    # ========================================================

    st.subheader(
        "Actual vs Predicted Traffic"
    )

    evaluation_group = st.selectbox(
        "Entry Point",
        list(GROUP_MAPPING.keys()),
        key="evaluation_group",
        accept_new_options=False
    )

    evaluation_group_id = (
        GROUP_MAPPING[
            evaluation_group
        ]
    )

    group_evaluation = evaluation_filtered[
        evaluation_filtered[
            'group_id'
        ] == evaluation_group_id
    ].copy()

    comparison_chart = (
        group_evaluation[
            [
                'toll_10_minute_block',
                'predicted_volume',
                'actual_volume'
            ]
        ]
        .set_index(
            'toll_10_minute_block'
        )
    )

    st.line_chart(
        comparison_chart,
        x_label="Time",
        y_label="Traffic Volume"
    )


    # ========================================================
    # MAE BY ENTRY POINT
    # ========================================================

    st.subheader(
        "Average Error by Entry Point"
    )

    mae_pct_chart_data = (
        group_performance_filtered[
            [
                "group_name",
                "mae",
                "mae_pct_of_mean"
            ]
        ]
        .sort_values(
            "mae_pct_of_mean"
        )
    )

    mae_pct_chart = (
        alt.Chart(mae_pct_chart_data)
        .mark_bar()
        .encode(
            x=alt.X(
                "group_name:N",
                sort=alt.EncodingSortField(
                    field="mae_pct_of_mean",
                    order="ascending"
                ),
                title="Entry Point"
            ),
            y=alt.Y(
                "mae_pct_of_mean:Q",
                title="Mean Absolute Error (%)"
            ),
            tooltip=[
                alt.Tooltip(
                    "group_name:N",
                    title="Entry Point"
                ),
                alt.Tooltip(
                    "mae_pct_of_mean:Q",
                    title="MAE (%)",
                    format=".1f"
                ),
                alt.Tooltip(
                    "mae:Q",
                    title="MAE (vehicles)",
                    format=".1f"
                )
            ]
        )
    )

    st.altair_chart(
        mae_pct_chart,
        use_container_width=True
    )

    # st.bar_chart(
    #     mae_chart,
    #     x_label="Entry Point",
    #     y_label="Mean Absolute Error Percentage (MAE%)"
    # )


    # ========================================================
    # FORECAST HORIZON
    # ========================================================

    st.subheader(
        "Accuracy by Forecast Horizon"
    )

    evaluation_filtered[
        'horizon_day'
    ] = (
        evaluation_filtered[
            'forecast_horizon_days'
        ]
        .apply(np.ceil)
        .astype(int)
    )

    horizon_performance = (
        evaluation_filtered
        .groupby(
            'horizon_day',
            as_index=False
        )
        .agg(
            mae=(
                'absolute_error',
                'mean'
            )
        )
    )

    st.line_chart(
        horizon_performance,
        x='horizon_day',
        y='mae',
        x_label="Days Ahead",
        y_label="Mean Absolute Error"
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
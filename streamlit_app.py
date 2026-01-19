import streamlit as st
import folium
from streamlit_folium import st_folium

# page config
st.set_page_config(page_title="Lower Manhattan Dashboard", layout="wide")

# title
st.title("Lower Manhattan Interactive Map")

# Sidebar for controls
st.sidebar.header("Map Controls")
map_style = st.sidebar.selectbox(
    "Select Map Style",
    ["CartoDB Light Mode", "CartoDB Dark Mode", "OpenStreetMap"]
)

# Map style tiles
tiles_dict = {
    "OpenStreetMap": "OpenStreetMap",
    "CartoDB Light Mode": "CartoDB Positron",
    "CartoDB Dark Mode": "CartoDB Dark_Matter"
}

# Center map on Lower Manhattan
# Coordinates are approximately where City Hall is
m = folium.Map(
    location=[40.7100, -73.9800],
    zoom_start=13,
    tiles=tiles_dict[map_style]
)

# notable locations
locations = {
    "One World Trade Center": [40.7127, -74.0134],
    "Battery Park": [40.7033, -74.0170],
    "Wall Street": [40.7074, -74.0113],
    "Brooklyn Bridge": [40.7061, -73.9969],
    "City Hall": [40.7128, -74.0060]
}

for name, coords in locations.items():
    folium.Marker(
        coords,
        popup=name,
        tooltip=name,
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

# display map
st_folium(m, width=1200, height=600)

# additional info
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.write("This dashboard displays an intercative map of Lower Manhattan with some key landmarks.")
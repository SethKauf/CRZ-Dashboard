import streamlit as st
import folium
from streamlit_folium import st_folium

# page config
st.set_page_config(page_title="Lower Manhattan Dashboard", layout="wide")

is_mobile = st.checkbox("Mobile View", value=False, key="mobile_toggle",
                        help="Toggle if map is not centered correctly")

if is_mobile:
    center_coords = [40.7074, -73.9800]  # Adjusted for mobile - shifted east
    bounds = [[40.6950, -73.9950], [40.7200, -73.9450]]
else:
    center_coords = [40.7074, -74.0050]
    bounds = [[40.6950, -74.0200],[40.7200, -73.9900]]

# title
st.title("Lower Manhattan Interactive Map")

# Sidebar for controls
st.sidebar.header("Map Controls")
map_style = st.sidebar.selectbox(
    "Select Map Style",
    ["Light Mode", "Dark Mode", "OpenStreetMap"]
)

# Map style tiles
tiles_dict = {
    "OpenStreetMap": "OpenStreetMap",
    "Light Mode": "CartoDB Positron",
    "Dark Mode": "CartoDB Dark_Matter"
}

# Center map on Lower Manhattan
# Coordinates are approximately where City Hall is
m = folium.Map(
    location=center_coords,
    zoom_start=8,
    tiles=tiles_dict[map_style],
    prefer_canvas=True,
    control_scale=True
)

# force the map to center on load
m.fit_bounds(bounds)


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
st_folium(m, width=1200, height=600, returned_objects=[])

# additional info
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.write("This dashboard displays an intercative map of Lower Manhattan with some key landmarks.")
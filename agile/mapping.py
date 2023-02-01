# Mapping function for the webapp demo
# Ernest Son
# Sam Chanow

import pandas as pd
import proximitypyhash as pph
from streamlit_folium import folium_static
import folium

def data_map(container, data=None, lois=None):
    if data is None and lois is None:
        return
    data_size = len(lois.index) if data is None else len(data.index)
    if data_size > 0:
        first_point = lois.iloc[0] if data is None else data.iloc[0]
        lat = first_point.latitude
        long = first_point.longitude
        with container:
            m = folium.Map(location=[lat, long], zoom_start=10)
            if data is not None:
                # Each marker will have a nice description
                def add_datapoint(row):
                    lat = row.latitude
                    long = row.longitude
                    p = "Location: "
                    p += str(row.latitude)
                    p += ", "
                    p += str(row.longitude)
                    p += " Timestamp: "
                    p += str(row.datetime)
                    p += "advertiser ID: "
                    p += row.advertiser_id
                    M = folium.Marker([lat,long], popup = p)
                    M.add_to(m)
                data.apply(add_datapoint, axis=1)
            # Add the LOI raster overlay to map
            if lois is not None:
                def add_loi(row):
                    lat = row.latitude
                    long = row.longitude
                    C = folium.CircleMarker([lat,long], radius=30, popup="LOI")
                    C.add_to(m)
                lois.apply(add_loi, axis=1)
            st_data = folium_static(m, width=725)
    else:
        container.write("No Data Points Available")


# Mapping functions for the webapp demo
# Ernest Son
# Sam Chanow

import pandas as pd
import proximitypyhash as pph
from streamlit_folium import folium_static
import folium

def colocation_data_map(data, lois, container):
    if lois is None:
        container.write("Generate locations of interest first!")
    data_size = len(data.index)
    data_limit = 1000000
    if (data_size > 0) and (data_size <= data_limit):
        first_point = data.iloc[0]
        lat = first_point.latitude
        long = first_point.longitude
        with container:
            m = folium.Map(location=[lat, long], zoom_start=10)
            # Add the LOI raster overlay to map
            lois.apply(lambda row: folium.CircleMarker([row.latitude, row.longitude], radius=30, popup="LOI").add_to(m), axis=1)
            # Add each data point to the plot
            # Take this info and change the marker description
            data.apply(lambda row: folium.Marker([row.latitude, row.longitude], popup = "Location: " + str(row.latitude) + ", " + str(row.longitude) + " Timestamp: " + str(row.datetime) + "AdID: " + row.advertiser_id).add_to(m), axis=1)
            st_data = folium_static(m, width=725)
    elif data_size > data_limit:
        container.write("Too much data to display. Change query to see data.")
    else:
        container.write("No Data Points Available")

def loi_data_map(data, container):
    data_size = len(data.index)
    data_limit = 1000000
    if (data_size > 0) and (data_size <= data_limit):
        first_point = data.iloc[0]
        lat = first_point.latitude
        long = first_point.longitude
        with container:
            m = folium.Map(location=[lat, long], zoom_start=10)
            # Add the LOI raster overlay to map
            data.apply(lambda row: folium.CircleMarker([row.latitude, row.longitude], radius=30, popup="LOI").add_to(m), axis=1)
            st_data = folium_static(m, width=725)
    elif data_size > data_limit:
        container.write("Too much data to display. Change query to see data.")
    else:
        container.write("No Data Points Available")

def data_map(data, container):
    data_size = len(data.index)
    data_limit = 1000000
    if (data_size > 0) and (data_size <= data_limit):
        first_point = data.iloc[0]
        lat = first_point.latitude
        long = first_point.longitude
        with container:
            m = folium.Map(location=[lat, long], zoom_start=10)
            # Add each data point to the plot
            # Take this info and change the marker description
            data.apply(lambda row: folium.Marker([row.latitude, row.longitude], popup = "Location: " + str(row.latitude) + ", " + str(row.longitude) + " Timestamp: " + str(row.datetime) + "AdID: " + row.advertiser_id).add_to(m), axis=1)
            st_data = folium_static(m, width=725)
    elif data_size > data_limit:
        container.write("Too much data to display. Change query to see filtered data.")
    else:
        container.write("No Data Points Available")


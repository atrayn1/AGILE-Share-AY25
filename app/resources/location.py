#Functions and Utility for Location Based Querys
#Sam Chanow

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import proximitypyhash as pph
import pygeohash as gh
from streamlit_folium import st_folium
from streamlit_folium import folium_static
import folium

#Creates a map component and adds it to the given container
def create_map(data, lat, long, container):
    #Picked 10000 as our data limit fro now, this can be changed
    if (len(data.index) > 0) and (len(data.index) <= 10000):

        with container:
            m = folium.Map(location=[lat, long], zoom_start=16)

            #Add each data point to the plot
            #TODO Have function take in list of strings that should be included in marker
            #Take this info and change the marker description
            data.apply(lambda row: folium.Marker([row["latitude"], row["longitude"]], popup="Timestamp: " + row['datetime'] + "AdID: " + row["advertiser_id"]).add_to(m), axis=1)
            #components.html(plotAll(parsed_df, lat, long, 23, 'satellite'), height = 600, width = 1000)
            st_data = folium_static(m, width=725)

            #Now we display geohashes for test purposes
            container.write(data)
    elif len(data.index) > 10000:
        container.write("Too much data to display. Change query to see filtered data.")
    else:
        container.write("No Data Points Available")


#function that query's a location and returns a filtered dataframe
def query_location(lat, long, radius, df):

    #First we need to check that all the fields were filled out and then cast to floats
    if lat == '' or long == '' or radius == '':
        return
    
    #convert to floats
    lat = float(lat)
    long = float(long)
    radius = float(radius)

    #We need to get the geohashes in the given radius at the given point
    geohashes = pph.get_geohash_radius_approximation(
        latitude=lat,
        longitude=long,
        radius=radius,
        precision=8,
        georaptor_flag=False,
        minlevel=1,
        maxlevel=12)
    
    #Now create the new smaller DF
    parsed_df = df.loc[df.geohash.isin(geohashes)]
    return parsed_df
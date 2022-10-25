#Simple streamlit geospacial query app
#Sam Chanow

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import proximitypyhash as pph
import pygeohash as gh
from streamlit_folium import st_folium
from streamlit_folium import folium_static
import folium

#function that query's a location and p
def query_location(lat, long, radius, container, df):

    #First we need to check that all the fields were filled out and then cast to floats
    #if lat == '' or long == '' or radius == '':
    #    return
    
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
    if len(parsed_df.index) > 0:
        #Now we display geohashes for test purposes
        container.write(parsed_df)

        with container:
            m = folium.Map(location=[lat, long], zoom_start=16)

            #Add each data point to the plot
            #Need to make this non-iterative later TODO
            parsed_df.apply(lambda row: folium.Marker([row["latitude"], row["longitude"]], popup="AdID: " + row["advertiser_id"]).add_to(m), axis=1)
            #components.html(plotAll(parsed_df, lat, long, 23, 'satellite'), height = 600, width = 1000)
            st_data = folium_static(m, width=725)

    else:
        container.write("No Data Points Available")

###App UI start###

df = pd.read_csv(
  "data/test_location_data_gh.csv"
)

st.title("AGILE")
st.header("Advertising and Geolocation Information Logical Extractor")

st.write(df.head())

#Simple dataset Query system
query_c = st.container()
query_c.header("Dataset Query")
#We need lat, long, radius
lat = query_c.text_input("Latitude")
long = query_c.text_input("Longitude")
radius = query_c.text_input("Radius")

query_c.button("Query", on_click=query_location, args=[lat, long, radius, query_c, df])
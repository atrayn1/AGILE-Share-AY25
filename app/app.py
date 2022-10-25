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

#Our function imports
import resources.location as loc 

def create_location_query(df, lat, long, radius, container):
    data = loc.query_location(lat, long, radius, df)
    loc.create_map(data, lat, long, container)

###App UI start###

df = pd.read_csv(
  "data/test_location_data_gh.csv"
)

st.title("AGILE")
st.subheader("Advertising and Geolocation Information Logical Extractor")

st.write("Dataset Sample")
st.write(df.head())

#Simple dataset Query system
query_c = st.expander("Location Query")
#We need lat, long, radius
lat = query_c.text_input("Latitude")
long = query_c.text_input("Longitude")
radius = query_c.text_input("Radius")

#Button calls geolocation query on click
query_c.button("Query", on_click=create_location_query, args=[df, lat, long, radius, query_c])
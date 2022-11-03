#Functions and Utility for AdId based querys
#Sam Chanow

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import proximitypyhash as pph
import pygeohash as gh
from streamlit_folium import st_folium
from streamlit_folium import folium_static
import folium

import resources.location as loc 

#Fuction that querys a dataframe and filters based on AdId
def query_adid(adid, df):
    if adid == '':
        return

    #Parse the df based on advertsing id
    parsed_df = df.loc[df.advertiser_id == adid]
    return parsed_df

#AdId query
def create_adid_query(df, ad_id, container):
  data = query_adid(ad_id, df)
  #uses the first data point as the center lat and long
  loc.create_map(data, data.iloc[0]['latitude'], data.iloc[0]['longitude'], container)

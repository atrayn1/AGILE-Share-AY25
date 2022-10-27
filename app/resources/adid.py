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

#Fuction that querys a dataframe and filters based on AdId
def query_adid(adid, df):
    if adid == '':
        return

    #Parse the df based on advertsing id
    parsed_df = df.loc[df.advertiser_id == adid]
    return parsed_df


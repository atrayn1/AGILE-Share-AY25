#Functions and Utility for Date based Queries
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
import resources.adid as adid

#function that queries a df by date range
def query_date(start_date, start_time, end_date, end_time, df):

    #Combien dates and times together for optimal filtering
    start_date = pd.datetime.combine(start_date, start_time)
    end_date = pd.datetime.combine(end_date, end_time)

    if start_date == '' or end_date == '':
        return

    #convert string to datetime
    df['date'] = pd.to_datetime(df['datetime'])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    #Parse the df by datetime
    date_filter = (df['date'] >= start_date) & (df['date'] <= end_date)
    parsed_df = df.loc[date_filter]
    print(parsed_df.head())
    parsed_df['datetime'] = parsed_df['datetime'].astype(str)
    print(parsed_df.head())
    print(len(parsed_df.index))
    return parsed_df

#Creates map based on date range dataframe
def create_date_map(df, container):
    adid.create_adid_map(df, container)
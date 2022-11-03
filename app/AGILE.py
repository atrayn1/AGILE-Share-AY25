#Main landing page for AGILE program
#Sam Chanow
#2022

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
import resources.adid as adid

#Global Vars
#Honestly right now this is just for the data so all containers have access to the Data value (which when intiialized
# will be a dataframe)
data = None
uploaded = False

###MAIN APP UI SETUP###
###THIS IS FOR FORMATTING ONLY###
###ACTIONS SHOULD OCCUR BELOW THIS SECTION###
###THIS IS EFFECTIVELY A STATIC "DEFAULT" PAGE###

#Title container
title_c = st.container()

#Title Columns
title_left, title_center = title_c.columns([1, 3])

title_center.title("AGILE")
title_center.subheader("Advertising and Geolocation Information Logical Extractor")
#Logo Image
title_left.image("images/logo.png") 

#Main page sidebar
sidebar = st.sidebar
sidebar.title("Data Options")

#Data Upload container (This is only for dev purposes)
data_upload_c = sidebar.container()

#The data Prievew
preview_c = st.container()
preview_c.subheader("Data Preview")
    

#The data analysis/filtering resulst container
results_c = st.container()
results_c.subheader("Analysis")


###ACTIONS FOR THE UI###
###THIS IS THE DYANMIC SECTION OF THE WEB APP###


with sidebar:
    with data_upload_c:
        raw_data = st.file_uploader("Upload Data File")
        #If a file has been uploaded
        if raw_data:
            data = pd.read_csv(raw_data, sep=",")
            uploaded = True

    filtering_ex = st.expander("Data Filtering")
    analysis_ex = st.expander("Data Analysis")

    #Filtering Expander
    with filtering_ex:
        adid_filter_c = st.container()
        with adid_filter_c:
            st.subheader("Ad Id Filtering")
            adid_form = st.form(key="adid_filter")
            with adid_form:
                ad_id = st.text_input("Advertiser ID")
                submitted = st.form_submit_button("Query")

                #What occurs when the form is submitted
                if submitted:
                    adid.create_adid_query(data, ad_id, results_c)

        st.subheader("Location Filtering")
        st.subheader("Time Filtering")

    #Analysis Expander
    with analysis_ex:
        st.subheader("KMeans?")
        st.subheader("Next Event Prediction?")
        st.subheader("Outlier Prediction?")
        st.subheader("Open source??")

#Preview container
with preview_c:
    #If Data means if they have uploaded a file
    if uploaded:
        st.dataframe(data.head())
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
import resources.date as date
import resources.overpassQuery as opq
import resources.loi as loi

#Global Vars
#Honestly right now this is just for the data so all containers have access to the Data value (which when intiialized
# will be a dataframe)

#Some session state variables that need to be maintained between reloads
if 'data' not in st.session_state:
    st.session_state.data = None

if 'uploaded' not in st.session_state:
    st.session_state.uploaded = False

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
filtering_ex = sidebar.expander("Data Filtering")
analysis_ex = sidebar.expander("Data Analysis")
reset_ex = sidebar.expander("Reset Data")

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
        #If a file has not yet been uploaded (this allows multiple form requests in unison)
        if raw_data and not st.session_state.uploaded:
            st.session_state.data = pd.read_csv(raw_data, sep=",")
            st.session_state.uploaded = True

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
                    st.session_state.data = adid.query_adid(ad_id, st.session_state.data) #Filter the data
                    adid.create_adid_map(st.session_state.data, results_c)

        location_filter_c = st.container()
        with location_filter_c:
            st.subheader("Location Filtering")
            location_form = st.form(key="location_filter")
            with location_form:
                #We need lat, long, radius
                lat = st.text_input("Latitude")
                long = st.text_input("Longitude")
                radius = st.text_input("Radius")
                submitted = st.form_submit_button("Query")

                #When the form is submitted
                if submitted:
                    st.session_state.data = loc.query_location(lat, long, radius, st.session_state.data)
                    loc.create_map(st.session_state.data, lat, long, results_c)

        time_filter = st.container()
        with time_filter:
            st.subheader("Time Filtering")
            time_form = st.form(key="time_filter")
            with time_form:
                start_date = st.date_input("Start Date")
                start_time = st.time_input("Time:", key="starttime")
                end_date = st.date_input("End Date")
                end_time = st.time_input("Time:", key="endtime")
                submitted = st.form_submit_button("Query")

                if submitted:
                    st.session_state.data = date.query_date(start_date, start_time, end_date, end_time, st.session_state.data)
                    date.create_date_map(st.session_state.data, results_c)

    #Analysis Expander
    with analysis_ex:
        overpass_analysis = st.container()
        with overpass_analysis:
            st.subheader("Overpass Query") #This will be an Overpass API integration
            overpass_form = st.form(key="overpass_adid")

            with overpass_form:
                ad_id = st.text_input("Advertiser ID")
                radius = st.text_input("Radius")
                submitted = st.form_submit_button("Query")

                #What occurs when the form is submitted
                if submitted:
                    st.session_state.data = adid.query_adid(ad_id, st.session_state.data) #Filter the data
                    res = opq.overpassPolyLineNearbyQuery(adid.query_adid(ad_id, st.session_state.data), radius)
                    results_c.write(res)

        loi_analysis = st.container()
        with loi_analysis:
            st.subheader("Location of Interest")
            loi_form = st.form(key="loi_form")

            with loi_form:
                ad_id = st.text_input("Advertiser ID")
                submitted = st.form_submit_button("Query")

                if submitted:
                    #We need to filter by adid and then perfrom loi analysis
                    #then we need to make a map
                    data = adid.query_adid(ad_id, st.session_state.data)
                    loi_data = loi.LOI(data)
                    #Here we need to make a map and pass the optional parameter for these location points
                    loc.create_map(data, data.iloc[0]['latitude'], data.iloc[0]['longitude'], results_c, loi_data=loi_data)


        st.subheader("KMeans?")
        st.subheader("Next Event Prediction?")
        st.subheader("Outlier Prediction?")
        st.subheader("Open source??")
    
    with reset_ex:
        reset_c = st.container()
        with reset_c:
            st.subheader("Reset")
            reset_form = st.form(key="reset")
            with reset_form:
                submitted = st.form_submit_button("RESET")

                #This will reset the state variable resetting the data to uploaded state
                if submitted:
                    st.session_state.data = pd.read_csv(raw_data, sep=",")

#Preview container
with preview_c:
    #If Data means if they have uploaded a file
    if st.session_state.uploaded:
        st.dataframe(st.session_state.data.head())
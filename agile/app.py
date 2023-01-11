# Main landing page for AGILE program
# Ernest Son
# Sam Chanow
# 2022-2023

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import proximitypyhash as pph
import pygeohash as gh
from streamlit_folium import st_folium
from streamlit_folium import folium_static
import folium

from filtering import query_location
from filtering import query_date
from filtering import query_adid
from mapping import data_map 
from mapping import loi_data_map
from mapping import colocation_data_map
from locations import locations_of_interest
from people import colocation
from utils.tag import polyline_nearby_query
from utils.geocode import reverse_geocode
from profile import Profile
from report import Report

# Some session state variables that need to be maintained between reloads
if 'data' not in st.session_state:
    st.session_state.data = None

if 'loi_data' not in st.session_state:
    st.session_state.loi_data = None

if 'uploaded' not in st.session_state:
    st.session_state.uploaded = False

# Title container
title_c = st.container()
title_left, title_center = title_c.columns([1, 3])
title_center.title('AGILE')
title_center.subheader('Advertising and Geolocation Information Logical Extractor')

# Logo Image
title_left.image('../images/new_logo.png')

# Main page sidebar
sidebar = st.sidebar
sidebar.title('Data Options')

# Data Upload container (This is only for dev purposes)
data_upload_sb = sidebar.container()
report_sb = sidebar.container()
filtering_ex = sidebar.expander('Data Filtering')
analysis_ex = sidebar.expander('Data Analysis')

# The data preview
preview_c = st.container()
preview_c.subheader('Data Preview')

# The data analysis/filtering results container
results_c = st.container()
results_c.subheader('Analysis')

# Dynamic content
with sidebar:

    # Upload data
    with data_upload_sb:
        raw_data = st.file_uploader('Upload Data File')
        # If a file has not yet been uploaded (this allows multiple form requests in unison)
        if raw_data and not st.session_state.uploaded:
            st.session_state.data = pd.read_csv(raw_data, sep=',')
            st.session_state.uploaded = True
        reset_c = st.container()
        with reset_c:
            reset_form = st.form(key='reset')
            with reset_form:
                # This will reset the state variable resetting the data to uploaded state
                if st.form_submit_button('RESET DATA'):
                    st.session_state.data = pd.read_csv(raw_data, sep=',')

    # Generate Report
    with report_sb:
        report_c = st.container()
        with report_c:
            report_form = st.form(key='report')
            with report_form:
                ad_id = st.text_input('Advertiser ID')
                prec = st.slider('Precision', min_value=1, max_value=12, value=10)
                exth = st.slider('Extended Stay Duration', min_value=1, max_value=24, value=7)
                reph = st.slider('Time Between Repeat Visits', min_value=1, max_value=72, value=24)
                colh = st.slider('Colocation Duration', min_value=1, max_value=24, value=2)
                report_button = st.form_submit_button('Generate Report')
                if report_button:
                    if st.session_state.uploaded:
                        device = Profile(st.session_state.data, ad_id, prec, exth, reph, colh)
                        Report(device)
                        results_c.write('Report generated!')
                    else:
                        results_c.write('Upload data first!')

    # Data Filtering Expander
    with filtering_ex:

        # Filter by advertising ID
        adid_filter_c = st.container()
        with adid_filter_c:
            st.subheader('Advertising ID Filtering')
            adid_form = st.form(key='adid_filter')
            with adid_form:
                ad_id = st.text_input('Advertiser ID')
                if st.form_submit_button('Query'):
                    st.session_state.data = query_adid(ad_id, st.session_state.data)
                    data_map(st.session_state.data, results_c)

        # Filter by lat/long
        location_filter_c = st.container()
        with location_filter_c:
            st.subheader('Location Filtering')
            location_form = st.form(key='location_filter')
            with location_form:
                # We need lat, long, and radius
                lat = st.text_input('Latitude')
                long = st.text_input('Longitude')
                radius = st.text_input('Radius')
                if st.form_submit_button('Query'):
                    st.session_state.data = query_location(lat, long, radius, st.session_state.data)
                    data_map(st.session_state.data, results_c)

        # Filter by timestamp
        time_filter = st.container()
        with time_filter:
            st.subheader('Time Filtering')
            time_form = st.form(key='time_filter')
            with time_form:
                start_date = st.date_input('Start Date')
                start_time = st.time_input('Time:', key='starttime')
                end_date = st.date_input('End Date')
                end_time = st.time_input('Time:', key='endtime')
                if st.form_submit_button('Query'):
                    st.session_state.data = query_date(start_date, start_time, end_date, end_time, st.session_state.data)
                    data_map(st.session_state.data, results_c)

    # Data Analysis Expander
    with analysis_ex:

        # Overpass API polyline
        overpass_analysis = st.container()
        with overpass_analysis:
            st.subheader('Overpass Query') # This will be an Overpass API integration
            overpass_form = st.form(key='overpass_adid')
            with overpass_form:
                ad_id = st.text_input('Advertiser ID')
                radius = st.text_input('Radius')
                if st.form_submit_button('Query'):
                    st.session_state.data = query_adid(ad_id, st.session_state.data) # Filter the data
                    res = polyline_nearby_query(query_adid(ad_id, st.session_state.data), radius)
                    results_c.write(res)

        # Locations of interest
        loi_analysis = st.container()
        with loi_analysis:
            st.subheader('Locations of Interest')
            loi_form = st.form(key='loi_form')
            with loi_form:
                loi_data = None
                ad_id = st.text_input('Advertiser ID')
                prec = st.slider('Precision', min_value=1, max_value=12, value=10)
                exth = st.slider('Extended Stay Duration', min_value=1, max_value=24, value=7)
                reph = st.slider('Time Between Repeat Visits', min_value=1, max_value=72, value=24)
                if st.form_submit_button('Query'):
                    # We need to filter by adid and then perform loi analysis
                    data = st.session_state.data
                    loi_data = locations_of_interest(data, ad_id, precision=prec, extended_duration=exth, repeated_duration=reph)
                    st.session_state.loi_data = loi_data
                    # Here we need to make a map and pass the optional parameter for these location points
                    loi_data_map(loi_data, results_c)
                    # Write Locations of Interest to the results container
                    results_c.write('Location of Interest Data')
                    results_c.write(loi_data)

        # Colocation
        colocation_analysis = st.container()
        with colocation_analysis:
            st.subheader('Colocation')
            colocation_form = st.form(key='colocation_form')
            with colocation_form:
                search_time = st.slider('Search Time', min_value=1, max_value=12, value=2)
                if st.form_submit_button('Query'):
                    data = st.session_state.data
                    loi_data = st.session_state.loi_data
                    colocation_data = colocation(data, loi_data, duration=search_time)
                    colocation_data_map(colocation_data, loi_data, results_c)
                    results_c.write('Colocation Data')
                    results_c.write(colocation_data)

# Preview container
with preview_c:
    # If Data means if they have uploaded a file
    if st.session_state.uploaded:
        st.dataframe(st.session_state.data)


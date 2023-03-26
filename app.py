# Main landing page for AGILE program
# Ernest Son
# Sam Chanow
# 2022-2023

import streamlit as st
import streamlit.components.v1 as components
import csv
import string
import pandas as pd
from datetime import datetime as dt
#import proximitypyhash as pph
#import pygeohash as gh
from streamlit_folium import st_folium
from streamlit_folium import folium_static
import folium
import numpy as np

from agile.filtering import query_location, query_date, query_adid, query_node
from agile.mapping import data_map 
from agile.locations import locations_of_interest
from agile.people import colocation
from agile.prediction import double_cluster, get_top_N_clusters
from agile.utils.tag import find_all_nearby_nodes
from agile.utils.geocode import reverse_geocode
from agile.utils.files import find
from agile.profile import Profile
from agile.report import Report
from agile.centrality import compute_top_centrality

from streamlit_option_menu import option_menu

# Make use of the whole screen
st.set_page_config(layout="wide")

# Some session state variables that need to be maintained between reloads
if 'data' not in st.session_state:
    st.session_state.data = None
if 'loi_data' not in st.session_state:
    st.session_state.loi_data = None
if 'uploaded' not in st.session_state:
    st.session_state.uploaded = False

# Replace Sidebar with data options Menu
# This will equal the value of the string selected
nav_bar = option_menu(None, ['Data', 'Filtering', 'Locations', 'Algorithms', 'Report'],
                      icons=['file-earmark-fill', 'funnel-fill', 'pin-map-fill', 'layer-forward', 'stack'],
                      menu_icon="cast", default_index=0, orientation="horizontal",
                    )

# Title container
title_c = st.container()
title_left, title_center = title_c.columns([1, 3])
title_center.title('AGILE')
title_center.subheader('Advertising and Geolocation Information Logical Extractor')

# Logo Image
#title_left.image(find('AGILE_Black.png', '/'))
# Path relative to files.py
title_left.image(find('../images/AGILE_Black.png'))

# Main page sidebar
sidebar = st.sidebar

# The data preview
preview_c = st.container()
preview_c.subheader('Total Data Preview')

# The data analysis/filtering results container
results_c = st.container()
results_c.subheader('Analysis')

# Based on what option is selected on the Nav Bar, a different container/expander will be displayed in the sidebar
if nav_bar == 'Data':
    # Data Upload container (This is only for dev purposes)
    sidebar.title('Data Options')
    data_upload_sb = sidebar.container()

    # Upload data
    relevant_features = ['geohash', 'datetime', 'latitude', 'longitude', 'advertiser_id']
    with data_upload_sb:
        raw_data = st.file_uploader('Upload Data File')
        # If a file has not yet been uploaded (this allows multiple form requests in unison)
        if raw_data and not st.session_state.uploaded:
            try:
                st.session_state.data = pd.read_csv(raw_data, sep=',', usecols=relevant_features)
                st.session_state.uploaded = True
            except:
                results_c.write('Invalid file format. Please upload a valid .csv file.')
        reset_c = st.container()
        with reset_c:
            reset_form = st.form(key='reset')
            with reset_form:
                # This will reset the state variable resetting the data to uploaded state
                if raw_data:
                    if st.form_submit_button('RESET DATA'):
                        try:
                            st.session_state.data = pd.read_csv(raw_data, sep=',', usecols=relevant_features)
                            st.session_state.uploaded = True
                        except:
                            results_c.write('Invalid file format. Please upload a valid .csv file.')


elif nav_bar == 'Filtering':

    filtering_ex = sidebar.container() #'Filtering'

    with filtering_ex:

        # Filter by advertising ID
        adid_filter_c = st.container()
        with adid_filter_c:
            st.subheader('Advertising ID Filtering')
            adid_form = st.form(key='adid_filter')
            with adid_form:
                adid = st.text_input('Advertiser ID')
                if st.form_submit_button('Query'):
                    st.session_state.data = query_adid(adid, st.session_state.data)
                    data_map(results_c, data=st.session_state.data)
                    results_c.write('Datapoints for ' + adid + ':')
                    results_c.write(st.session_state.data)

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
                    data_map(results_c, data=st.session_state.data)
                    results_c.write('Datapoints around ' + lat + ', ' + long + ' in a radius of ' + radius + ' meters:')
                    results_c.write(st.session_state.data)

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
                start = str(dt.combine(start_date, start_time))
                end = str(dt.combine(end_date, end_time))
                if st.form_submit_button('Query'):
                    st.session_state.data = query_date(start_date, start_time, end_date, end_time, st.session_state.data)
                    data_map(results_c, data=st.session_state.data)
                    results_c.write('Datapoints between ' + start + ' and ' + end + ':')
                    results_c.write(st.session_state.data)

elif nav_bar == 'Locations':
    locations_ex = sidebar.container() #'Locations'

    with locations_ex:

        # Overpass specific node query
        node_analysis = st.container()
        with node_analysis:
            st.subheader('Node Query')
            node_form = st.form(key='node')
            with node_form:
                lat = st.text_input('Latitude')
                long = st.text_input('Longitude')
                radius = st.text_input('Radius')
                node = st.text_input('Node')
                if st.form_submit_button('Query'):
                    node_data = query_node(lat, long, radius, node)
                    data_map(results_c, data=node_data)
                    results_c.write(node + ' found around ' + lat + ', ' + long + ' within a radius of ' + radius + ' meters:')
                    results_c.write(node_data)

        # Centrality analysis
        centrality_analysis = st.container()
        with centrality_analysis:
            st.subheader('Location Centrality Query')
            centrality_form = st.form(key='centrality')
            with centrality_form:
                lat = st.text_input('Latitude')
                long = st.text_input('Longitude')
                radius = st.text_input('Radius')
                if st.form_submit_button('Query'):
                    centrality_data = compute_top_centrality(lat, long, radius, 5, st.session_state.data)
                    data_map(results_c, lois=centrality_data)
                    results_c.write('The locations with the highest centrality to the AdIDs at the entered location are:')
                    results_c.write(centrality_data)

        # Overpass API polyline
        overpass_analysis = st.container()
        with overpass_analysis:
            st.subheader('Overpass Polyline Query') # This will be an Overpass API integration
            overpass_form = st.form(key='polyline')
            with overpass_form:
                adid = st.text_input('Advertiser ID')
                radius = st.text_input('Radius')
                if st.form_submit_button('Query'):
                    st.session_state.data = query_adid(adid, st.session_state.data) # Filter the data
                    res = find_all_nearby_nodes(st.session_state.data, radius)
                    results_c.write(res)

elif nav_bar == 'Algorithms':
    algorithms_ex = sidebar.container() #'Algorithms'

    with algorithms_ex:

        # (Clustering) locations of interest
        cluster_analysis = st.container()
        with cluster_analysis:
            st.subheader('Top Clusters')
            cluster_form = st.form(key='cluster_form')
            with cluster_form:
                loi_data = None
                adid = st.text_input('Advertiser ID')
                num_clusters = st.slider('Number of Clusters', min_value=1, max_value=10, value=4)
                if st.form_submit_button('Query'):
                    # We need to filter by adid and then perform loi analysis
                    data = st.session_state.data
                    cluster_data = double_cluster(adid, data)
                    loi_data = get_top_N_clusters(cluster_data, num_clusters)
                    if loi_data is None:
                        results_c.write('No Clusters Found')
                    else:
                        st.session_state.loi_data = loi_data
                        # Here we need to make a map and pass the optional parameter for these location points
                        data_map(results_c, lois=st.session_state.loi_data)
                        # Write Locations of Interest to the results container
                        results_c.write('Cluster Data')
                        results_c.write(loi_data)

        # (Traditional) locations of interest
        loi_analysis = st.container()
        with loi_analysis:
            st.subheader('Locations of Interest')
            loi_form = st.form(key='loi_form')
            with loi_form:
                loi_data = None
                adid = st.text_input('Advertiser ID')
                ext_h = st.slider('Extended Stay Duration', min_value=1, max_value=24, value=7)
                rep_h = st.slider('Time Between Repeat Visits', min_value=1, max_value=72, value=24)
                if st.form_submit_button('Query'):
                    # We need to filter by adid and then perform loi analysis
                    data = st.session_state.data
                    loi_data = locations_of_interest(data, adid, ext_h, rep_h)
                    st.session_state.loi_data = loi_data
                    # Here we need to make a map and pass the optional parameter for these location points
                    data_map(results_c, lois=st.session_state.loi_data)
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
                    data_map(results_c, data=colocation_data, lois=loi_data)
                    results_c.write('Colocation Data')
                    results_c.write(colocation_data)

        # Prediction
        pred_analysis = st.container()
        with pred_analysis:
            st.subheader('Movement Prediction')
            pred_form = st.form('pred')
            with pred_form:
                adid = st.text_input('Advertiser ID')
                start_time = st.time_input('Time:', key='time')
                start_day = st.slider('Day:', min_value=0, max_value=6, value=2)
                if st.form_submit_button('Predict'):
                    st.session_state.profile = Profile(st.session_state.data, adid)
                    if not st.session_state.profile.model_trained():
                        st.session_state.profile.model_train()
                    # Convert the time input to a datetime
                    # str(dt.combine(start_date, start_time))
                    # Using an arbitary date because this algorith monly cares about the time of day
                    start_time = pd.to_datetime(str(dt.combine(pd.to_datetime('2018-01-01'), start_time)))
                    start_time = np.array((start_time - start_time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()).reshape(-1, 1)
                    result_label, result_centroid = st.session_state.profile.model_predict(start_time, start_day)
                    data_map(results_c, lois=result_centroid)
elif nav_bar == 'Report':
    report_sb = sidebar.container() #'Report'
    with report_sb:
        report_c = st.container()
        with report_c:
            report_form = st.form(key='report')
            with report_form:
                adid = st.text_input('Advertiser ID')
                exth = st.slider('Extended Stay Duration', min_value=1, max_value=24, value=7)
                reph = st.slider('Time Between Repeat Visits', min_value=1, max_value=72, value=24)
                colh = st.slider('Colocation Duration', min_value=1, max_value=24, value=2)
                report_button = st.form_submit_button('Generate Report')
                if report_button:
                    if st.session_state.uploaded:
                        device = Profile(st.session_state.data, adid, exth, reph, colh)
                        Report(device)
                        results_c.write('Report generated!')
                    else:
                        results_c.write('Upload data first!')
else:
    pass #Nothing should happen, it should never be here

# Dynamic content
#with sidebar:
    # Data Filtering Expander
    # Analysis Expander

    # Generate Report


# Preview container
with preview_c:
    # If Data means if they have uploaded a file
    if st.session_state.uploaded:
        st.dataframe(st.session_state.data)


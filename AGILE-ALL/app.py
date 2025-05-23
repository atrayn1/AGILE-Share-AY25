# Main landing page for AGILE program
# Ernest Son
# Sam Chanow
# 2022-2023

import streamlit as st
import streamlit.components.v1 as components
import csv
import string
import os
import pickle
import pandas as pd
from datetime import datetime as dt
from base64 import b64encode
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
from agile.utils.files import find, random_line, save, random_name, generate_aliases
from agile.utils.dataframes import modify_and_sort_columns, clean_and_verify_columns
from agile.profile import Profile
from agile.report import Report
from agile.centrality import compute_top_centrality
from agile.overview import adid_value_counts

# AY 25 Addition
from agile.graphing import createGraph, connectNodes, connectCurrentNodes, expandNode, addADID, find_cliques_for_adid

from streamlit_option_menu import option_menu
import pygeohash as gh

from visual_graph import generate_visualization

# Make use of the whole screen
st.set_page_config(layout="wide")

# Some session state variables that need to be maintained between reloads
if 'data' not in st.session_state:
    st.session_state.data = None
if 'loi_data' not in st.session_state:
    st.session_state.loi_data = None
if 'uploaded' not in st.session_state:
    st.session_state.uploaded = False
if 'file_source' not in st.session_state:
    # Iterate through all files in the directory and deleted the saved ones
    if os.path.exists('./saved_data'):
        for filename in os.listdir(os.path.abspath('./saved_data')):
            file_path = os.path.join(os.path.abspath('./saved_data'), filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    st.session_state.file_source = False
if 'coloc_ids' not in st.session_state:
    st.session_state.coloc_ids = pd.DataFrame(columns=['Colocated ADIDs','Alias'])
if 'generated_reports' not in st.session_state:
    st.session_state.generated_reports = pd.DataFrame(columns=['ADID', 'Alias','Profile'])
if 'alias_ids' not in st.session_state:
    st.session_state.alias_ids = {}
if 'main_adid' not in st.session_state:
    st.session_state.main_adid = ""       # the adid you want to further explore



# Replace Sidebar with data options Menu
# This will equal the value of the string selected

# Title container
title_c = st.container()
title_img, title_left, space = title_c.columns([1, 3, 1])

title_left.markdown("<h2 style='text-align: center; color: black;'>AGILE</h2>", unsafe_allow_html=True)

title_left.markdown("<h3 style='text-align: center; color: black;'>Advertising and Geolocation Information Logical Extractor </h3>", unsafe_allow_html=True)

#title_left.title('AGILE')
#title_left.subheader('Advertising and Geolocation Information Logical Extractor')
#data_reset_button = title_left.button('Reset Data')
#keep_aliases_check = title_left.checkbox('Keep Aliases', True)


title_img.image(find('../img/AGILE_Black.png'), width = 180)


nav_bar = option_menu(None, ['Data', 'Graph', 'Filtering', 'Locations', 'Algorithms', 'Report'],
                      icons=['file-earmark-fill', 'funnel-fill', 'pin-map-fill', 'layer-forward', 'stack', 'command'],
                      menu_icon="cast", default_index=0, orientation="horizontal",
                    )



# Logo Image
#title_left.image(find('AGILE_Black.png', '/'))
# Path relative to files.py
#title_left.image(find('../img/AGILE_Black.png'), width = 200)

# Main page sidebar
sidebar = st.sidebar

# The data preview
blank = st.container()
blank.subheader('')

data_opts = st.container()
data_reset_button = data_opts.button('Reset Data')
keep_aliases_check = True #data_opts.checkbox('Keep Aliases', True)

preview_c = st.container()
preview_c.subheader('Total Data Preview')

overview_c = st.container()
overview_c.subheader('Data Overview')

# The data analysis/filtering results container
results_c = st.container()
results_c.subheader('Analysis')


if not os.path.exists('saved_data'):
    os.makedirs('saved_data')

# Based on what option is selected on the Nav Bar, a different container/expander will be displayed in the sidebar
if nav_bar == 'Data':
    # Data Upload container (This is only for dev purposes)
    sidebar.title('Data')
    sidebar.write("The modules below import data from a file/database into the app.")
    data_upload_sb = sidebar.container()

    # Upload data
    #relevant_features = ['datetime', 'latitude', 'longitude', 'advertiser_id']
    with data_upload_sb:
        raw_data = st.file_uploader('Upload Data File')

        # adding 'Demo' button 
        demo_button = st.button('Load Demo Data')

        if demo_button:
            demo_file_path = 'data/houthi_adid_dataset.csv' 

            # Check if the demo CSV file exists in the current directory
            if os.path.exists(demo_file_path):
                # Load the demo data from the CSV file
                st.session_state.data = pd.read_csv(demo_file_path, sep=',')
                st.session_state.uploaded = True
                st.session_state.file_source = demo_file_path

                # Process the demo data
                try:
                    # Clean and verify columns (ensure required columns exist)
                    st.session_state.data = clean_and_verify_columns(st.session_state.data)
                except:
                    preview_c.error('Error with modifying and sorting the columns. Please ensure your CSV file has the correct columns.')

                try:
                    # Convert 'datetime' column to pandas datetime type
                    st.session_state.data['datetime'] = pd.to_datetime(st.session_state.data['datetime'], errors='coerce')
                except:
                    results_c.error('Could not convert "datetime" column to pd.DateTime type.')

                # Generate geohashes if the column doesn't exist or isn't correct
                if not 'geohash' in st.session_state.data.columns or not len(st.session_state.data['geohash'].iloc[0]) == 10:
                    with st.spinner("Geohashing the data..."):
                        st.session_state.data['geohash'] = st.session_state.data.apply(lambda d: gh.encode(d.latitude, d.longitude, precision=10), axis=1)

                # Perform final preprocessing operations before displaying the data
                try:
                    st.session_state.data = modify_and_sort_columns(st.session_state.data)
                except:
                    results_c.error('Error with modifying and sorting the columns. Please ensure your CSV file has the correct columns.')
                
                # Generate aliases for ADIDs
                st.session_state.alias_ids = generate_aliases(st.session_state.data)

                # Check if all ADIDs were assigned an alias
                if 'Unnamed_Alias' in st.session_state.alias_ids.values():
                    preview_c.error("WARNING: Due to the amount of ADIDs in your data, not every ADID was assigned an alias.")

                # Save the data to a pickle file for future use
                with st.spinner("Saving the modified data locally for fast reaccessing..."):
                    save('saved_df.pkl', st.session_state.data)

                st.write("Demo Data Loaded Successfully")

            else:
                st.write("Demo file (houthi_adid_dataset.csv) not found.")

        # If a file has not yet been uploaded (this allows multiple form requests in unison)
        if raw_data and raw_data.name != st.session_state.file_source:
            st.session_state.data = pd.read_csv(raw_data, sep=',')
            st.session_state.uploaded = True
            st.session_state.file_source = raw_data.name
            
            # makes sure all the required columns are present (latitude, longitude, datetime, advertiser_id) and
            # cleans the names if they are close (Latitude instead of latitude)
            try:
                st.session_state.data = clean_and_verify_columns(st.session_state.data)
            except:
                preview_c.error('Error with modifying and sorting the columns. Please ensure you uploaded a csv file with advertiser_id, datetime, latitude and longitude columns.')
            
            try:
                st.session_state.data['datetime'] = pd.to_datetime(st.session_state.data['datetime'],errors='coerce')
            except:
                results_c.error('Could not convert "datetime" column to pd.DateTime type')
            

            # Check to make sure the uploaded data has geohashes
            # If it does not, generate them on the fly
            if not 'geohash' in st.session_state.data.columns or not len(st.session_state.data['geohash'].iloc[0]) == 10:
                # Something is wrong, either the column does not exist
                # Or it is the wrong precision geohash
                # So we generate it manually

                with st.spinner("Geohashing the data..."):
                    st.session_state.data['geohash'] = st.session_state.data.apply(lambda d : gh.encode(d.latitude, d.longitude, precision=10), axis=1)
        
            
            try:
                # perform final preprocessing operations before displaying the data
                # all the code for these next couple lines can be found in agile/utils/dataframes.py
                st.session_state.data = modify_and_sort_columns(st.session_state.data)
            except:
                results_c.error('Error with modifying and sorting the columns. Please ensure you uploaded a csv file with advertiser_id, datetime, latitude and longitude columns.')
            
         
            # this function generates a random alias for each ADID, though it saves it in a dictionary (st.session_state.alias_ids) 
            # because saving it to st.session_state.data would take a long time. If we ever want to access or modify
            # an alias, we use this dictionary
            st.session_state.alias_ids = generate_aliases(st.session_state.data)
            
            # the case where there are more ADIDs than generated aliases. This should never really happen! There are 1,368,896 possible generated aliases
            if 'Unnamed_Alias' in st.session_state.alias_ids.values():
                preview_c.error("WARNING: Due to the amount of ADIDs in your data, not every ADID was assigned an alias")
            
            
            # Save the data to a pickle file, located in the /saved_data directory
            # This is done so it can be reloaded with the "reset data" button
            with st.spinner("Saving the modified data locally for fast reaccessing..."):
                #save('original_df.pkl',st.session_state.data)  
                save('saved_df.pkl',st.session_state.data)
            
        # If there is a dataframe, update the "Data Overview," "Time Distribution," and "Geohash Distribution" statistics 
        if st.session_state.uploaded and not data_reset_button:
            # Data overview
            try:
                # Update the value counts for an ADID
                overview_c.dataframe(adid_value_counts(st.session_state.data), height=300)
            except:
                overview_c.error("Could not load overview statistics.")
            
            # Time distribution
            try:
                time_data = overview_c.container()
                time_data.subheader('Time Distribution')
                time_data.dataframe(pd.DataFrame(st.session_state.data['datetime']).describe())
            except:
                time_data.error('Could not load time statistics.')   
                
            # Geohash distribution
            try: 
                geohash_distro = overview_c.container()
                geohash_distro.subheader('Geohash Distribution')
                geohash_distro_data = st.session_state.data.groupby('geohash').size().reset_index(name='count').sort_values(by='count', ascending=False)
                geohash_distro.dataframe(geohash_distro_data)
            except:
                geohash_distro.error("Could not load geohash statistics.")
            
    # find generated alias for an ADID. This looks up some ADID's alias in the dictionary st.session_state.alias_ids
    alias_finder = sidebar.container()
    with alias_finder:
        st.subheader('Check the alias for an ADID')
        st.write('Aliases are randomly assigned to all ADIDs, even if the advertiser_id_alias column does not populate. Whether you enter a custom alias or look at the randomly generated alias, you can find it here.')
        
        alias_finder_form = st.form('find_alias')
        with alias_finder_form:
            #alias_form_text = st.text_input('Advertiser ID')
            if st.session_state.main_adid != '':
                alias_form_text = st.text_input('Advertiser ID', value=st.session_state.main_adid)
            else:
                alias_form_text = st.text_input('Advertiser ID')
            if st.form_submit_button('Find Alias'):
                try:
                    found_alias = st.session_state.alias_ids[alias_form_text.strip()]
                    st.info(found_alias)
                    st.session_state.data.loc[st.session_state.data['advertiser_id'] == alias_form_text, 'advertiser_id_alias'] = found_alias
                except:
                    st.info(f'ADID {alias_form_text} was not found')
            
    # Container for adding an alias to an ADID
    renamer = sidebar.container()
    with renamer:
        st.subheader('Add Alias for an ADID')
        st.write("Choose a name yourself or generate a random name for an ADID.\nNote: Though it's not displayed in the data to the right, each advertising ID is automatically assigned an alias to begin with. It is not updated in the DataFrame because it could take several minutes, but these names aliases still be seen in reports and across AGILE.")
        
        # Creates the form which will hold the text boxes, check box, and button
        rename_form = st.form('rename_adid')
        with rename_form:
            #adid_to_update = st.text_input('Advertiser ID')
            if st.session_state.main_adid != '':
                adid_to_update = st.text_input('Advertiser ID', value=st.session_state.main_adid)
            else:
                adid_to_update = st.text_input('Advertiser ID')
            new_name_text = st.text_input('Custom Name')
            
            if st.form_submit_button('Assign Name'):
                # Case where an invalid ADID is entered
                if adid_to_update.strip() not in st.session_state.data['advertiser_id'].values:
                    preview_c.error('Error: Invalid ADID. Please re-enter the ADID')
                # Case where the user enters nothing but clicks the button
                elif new_name_text == '':
                    preview_c.error('Error: Please enter at least one character for a custom name')
                # Case where the user enters an alias that is already being used
                elif (new_name_text in st.session_state.data['advertiser_id_alias'].values or new_name_text in st.session_state.alias_ids.values()):
                    preview_c.error(f'Error: The alias {new_name_text} is already in use')
                # No errors, change the current alias with the alias the user entered
                else:
                    with st.spinner('Adding Alias...'):
                        st.session_state.alias_ids[adid_to_update] = new_name_text
                        st.session_state.data.loc[st.session_state.data['advertiser_id'] == adid_to_update, 'advertiser_id_alias'] = new_name_text
                        save('saved_df.pkl',st.session_state.data)
                
                        #st.session_state.file_source = os.path.abspath('./saved_data/saved_df.pkl')

elif nav_bar == 'Filtering':
    
    sidebar.title('Filtering')

    filtering_ex = sidebar.container() #'Filtering'

    with filtering_ex:

        st.write("The modules below filter the data by Advertising ID, Location, or Time. These changes will remain until \
                 the data is reset in the 'Data' section.")

        # Filter by advertising ID4d02768a-0340-d327-4482-78a7a7420829
        adid_filter_c = st.container()
        with adid_filter_c:
            st.subheader('Advertising ID Filtering')

            st.info("Filter the data by advertising ID, i.e. a single device.")

            adid_form = st.form(key='adid_filter')
            with adid_form:
                print("main adid: ", st.session_state.main_adid)

                # Pre-fill the value with st.session_state.main_adid if it's not an empty string
                if st.session_state.main_adid != '':
                    adid = st.text_input('Advertiser ID', value=st.session_state.main_adid)
                else:
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
            st.info('Filter the data by location, given as latitude and longitude. The range is the radius around the latitude \
                     and longitude in which to filter the data.')
            location_form = st.form(key='location_filter')
            with location_form:
                # We need lat, long, and radius
                lat = st.text_input('Latitude')
                long = st.text_input('Longitude')
                radius = st.text_input('Radius (meters)')
                if st.form_submit_button('Query'):
                    st.session_state.data = query_location(lat, long, radius, st.session_state.data)
                    data_map(results_c, data=st.session_state.data)
                    results_c.write('Datapoints around ' + lat + ', ' + long + ' in a radius of ' + radius + ' meters:')
                    results_c.write(st.session_state.data)

        # Filter by timestamp
        time_filter = st.container()
        with time_filter:
            st.subheader('Time Filtering')
            st.info("Filter the data by timestamp.")
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

    sidebar.title('Locations')

    locations_ex = sidebar.container() #'Locations'

    with locations_ex:

        st.write("The modules below provide more information about a specific location in the data.")

        # Overpass specific node query
        node_analysis = st.container()
        with node_analysis:
            st.subheader('Node Query')
            st.info('Query the Overpass API for named locations within a certain circular radius of a given latitude and longitude.\
                     The radius is in meters. To search for a specific type of location, enter the location name into the Node field below.')
            node_form = st.form(key='node')
            with node_form:
                lat = st.text_input('Latitude')
                long = st.text_input('Longitude')
                radius = st.text_input('Radius (meters)')
                node = st.text_input('Node (name)')
                if st.form_submit_button('Query'):
                    with st.spinner(text="Computing..."):
                        node_data = query_node(lat, long, radius, node)
                        data_map(results_c, data=node_data)
                        results_c.write(node + ' found around ' + lat + ', ' + long + ' within a radius of ' + radius + ' meters:')
                        results_c.write(node_data)

        # Centrality analysis
        centrality_analysis = st.container()
        with centrality_analysis:
            st.subheader('Location Centrality Query')
            # TODO I think this description needs work from someone who is not me (sam)
            st.info('Determine the most visited locations for the advertising IDs found at the given latitude and longitude \
                     within a certain radius in meters.')
            centrality_form = st.form(key='centrality')
            with centrality_form:
                lat = st.text_input('Latitude')
                long = st.text_input('Longitude')
                radius = st.text_input('Radius (meters)')
                if st.form_submit_button('Query'):
                    with st.spinner(text="Computing..."):
                        centrality_data = compute_top_centrality(lat, long, radius, 5, st.session_state.data)
                        data_map(results_c, lois=centrality_data)
                        results_c.write('The locations with the highest centrality to the ADIDs at the entered location are:')
                        results_c.write(centrality_data)

        # Overpass API polyline
        overpass_analysis = st.container()
        with overpass_analysis:
            st.subheader('Overpass Polyline Query') # This will be an Overpass API integration
            st.info('Query the Overpass API for points of interest along the path of a single advertising ID. The radius is the \
                    circular radius around each point for the advertising ID to search for these points of interest. The radius is in meters.')
            overpass_form = st.form(key='polyline')
            with overpass_form:
                if st.session_state.main_adid != '':
                    adid = st.text_input('Advertiser ID', value=st.session_state.main_adid)
                else:
                    adid = st.text_input('Advertiser ID')
                radius = st.text_input('Radius (meters)')
                if st.form_submit_button('Query'):
                    with st.spinner(text="Computing..."):
                        st.session_state.data = query_adid(adid, st.session_state.data) # Filter the data
                        res = find_all_nearby_nodes(st.session_state.data, radius)
                        results_c.write(res)

elif nav_bar == 'Algorithms':

    sidebar.title('Algorithms')

    algorithms_ex = sidebar.container() #'Algorithms'

    with algorithms_ex:

        st.write("The modules below provide more information about a single advertiser ID (a single device) in the data.")

        # (Clustering) locations of interest
        cluster_analysis = st.container()
        with cluster_analysis:
            st.subheader('Top Clusters')
            st.info('Determine the top N locations of interest for a single Advertising ID. The top cluster is usually the \
                    home location, wheras the second and third cluster are usually work and frequently visited locations. Results vary \
                    based on data size and consistency.')
            cluster_form = st.form(key='cluster_form')
            with cluster_form:
                loi_data = None
                if st.session_state.main_adid != '':
                    adid = st.text_input('Advertiser ID', value=st.session_state.main_adid)
                else:
                    adid = st.text_input('Advertiser ID')
                num_clusters = st.slider('Number of Clusters', min_value=1, max_value=10, value=4)
                if st.form_submit_button('Query'):
                    with st.spinner(text="Computing..."):
                        # We need to filter by adid and then perform loi analysis
                        data = st.session_state.data
                        cluster_data = double_cluster(adid, data)

                        loi_data = get_top_N_clusters(cluster_data, num_clusters)

                        # Top N clusters clobbers the geohash so we need to calculate it for the centroids again
                        # loi_data['geohash'] = loi_data.apply(lambda d : gh.encode(d.latitude, d.longitude, precision=10), axis=1)
                        # loi_data['advertiser_id'] = adid

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
            st.info('Determine the top locations of interest for a single Advertising ID. This algorithm does not use AI/ML, \
                    it instead flags locations based on time spent at the location. The Extended Stay Duration is the length of time \
                    required at a single location (without leaving) to be flagged. The Time Between Repeat Visits is the lenght of time \
                    between two data points at the same location for them to be considered wo separate visits.')
            loi_form = st.form(key='loi_form')
            with loi_form:
                loi_data = None
                if st.session_state.main_adid != '':
                    adid = st.text_input('Advertiser ID', value=st.session_state.main_adid)
                else:
                    adid = st.text_input('Advertiser ID')
                ext_h = st.slider('Extended Stay Duration', min_value=1, max_value=24, value=7)
                rep_h = st.slider('Time Between Repeat Visits', min_value=1, max_value=72, value=24)
                if st.form_submit_button('Query'):
                    with st.spinner(text="Computing..."):
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
            st.info('Identify other advertising IDs that show up at any of the locations of interest within a certain time frame. \
                    One of the two algorithms above MUST be run before using this algorithm. Search Time is the allowable difference between \
                    Location of Interest datapoint and the other advertising IDs datapoint\'s time stamp')
            colocation_form = st.form(key='colocation_form')
            with colocation_form:
                search_time = st.slider('Search Time (hr)', min_value=1, max_value=12, value=2)
                if st.form_submit_button('Query'):
                    with st.spinner(text="Computing..."):
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
            st.info('Predict the location of an advertising ID at a certain day of the week, and time of day. One of the locations of Interest \
                    algorithms MUST be run before using this module.')
            pred_form = st.form('pred')
            with pred_form:
                if st.session_state.main_adid != '':
                    adid = st.text_input('Advertiser ID', value=st.session_state.main_adid)
                else:
                    adid = st.text_input('Advertiser ID')
                start_time = st.time_input('Time:', key='time')
                start_day = st.slider('Day (0 = Saturday, 6 = Sunday):', min_value=0, max_value=6, value=2)
                if st.form_submit_button('Predict'):
                    with st.spinner(text="Computing..."):
                        st.session_state.profile = Profile(st.session_state.data, adid)
                        if not st.session_state.profile.model_trained():
                            st.session_state.profile.model_train()
                        # Convert the time input to a datetime
                        # str(dt.combine(start_date, start_time))
                        # Using an arbitary date because this algorithm only cares about the time of day
                        start_time = pd.to_datetime(str(dt.combine(pd.to_datetime('2018-01-01'), start_time)))
                        start_time = np.array((start_time - start_time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()).reshape(-1, 1)
                        result_label, result_centroid = st.session_state.profile.model_predict(start_time, start_day)
                        data_map(results_c, lois=result_centroid)
            # TODO Give some visual or text showing the accuracy of the model that was trained
            try:
                st.write('Model Accuracy: ' + str(st.session_state.profile.model_accuracy * 100) + "%")
                st.progress(st.session_state.profile.model_accuracy)

                st.write('Model Reliability: ' + str(st.session_state.profile.reliability()))
                st.progress(st.session_state.profile.reliability())
            except:
                st.write('No Model Trained Yet')
                
elif nav_bar == 'Report':

    sidebar.title('Report')

    sidebar.write("The module below generates a report in PDF format about a single advertiser ID (a single device) in the data.")

    sidebar.subheader('Generate Report')
    
    report_sb = sidebar.container() #'Report'
    with report_sb:
        report_c = st.container()
        colocs = st.container()
        generated_reps = st.container()
 
        with report_c:
            report_form = st.form(key='report')
            with report_form:
                if st.session_state.main_adid != '':
                    adid = st.text_input('Advertiser ID', value=st.session_state.main_adid)
                else:
                    adid = st.text_input('Advertiser ID')
                #exth = st.slider('Extended Stay Duration', min_value=1, max_value=24, value=7)
                #reph = st.slider('Time Between Repeat Visits', min_value=1, max_value=72, value=24)
                #colh = st.slider('Colocation Duration', min_value=1, max_value=24, value=2)
                report_button = st.form_submit_button('Generate Report')

                #find the number of days this data covers to see if there is sufficient data for an adid
                st.session_state.data['datetime'] = pd.to_datetime(st.session_state.data['datetime'])
                min_date = st.session_state.data['datetime'].min()
                max_date = st.session_state.data['datetime'].max()
                days_covered = (max_date - min_date).days + 1

                # if the button is clicked...
                if report_button:
                    
                    # This block of code calculates whether or not there were a sufficient amount of datapoints for a thorough analysis.
                    # If there are not enough, then a disclaimer is shown that the algorithms may be inaccurate
                    if adid not in st.session_state.data['advertiser_id'].values:
                        results_c.error('ADID is invalid. Please enter a different ADID')
                    elif (adid_value_counts(st.session_state.data)['Occurences in Data'].get(adid) * 1.0 / days_covered) < 200:                        
                        suff_data = 0
                    else:
                        suff_data = 1
                       
                    if st.session_state.uploaded:

                        # If no report has been generated on this ADID already...
                        if adid not in st.session_state.generated_reports['ADID'].values:
                            
                            # Check if the adid has an alias added (it should)
                            if len(st.session_state.data.query('advertiser_id==@adid')['advertiser_id_alias'].unique()) > 0:
                                adid_alias = st.session_state.data.query('advertiser_id==@adid')['advertiser_id_alias'].unique()[0]
                            else:
                                adid_alias = st.session_state.alias_ids[adid]
                                
                            # set up the profile for the adid
                            # creating a profile also creates the LOI DataFrame, which may take a minute or two depending on the size of the data                
                            device = Profile(data=st.session_state.data, ad_id=adid,
                                            alias=adid_alias, sd = suff_data, alias_dict = st.session_state.alias_ids)
                            
                            # Update all of the aliases for this ADID in the main dataframe. 
                            # This is because all of the aliases are stored in a dictionary (st.session_state.alias_ids) rather than the main dataframe (st.session_state.data)
                            st.session_state.data.loc[st.session_state.data['advertiser_id'] == adid, 'advertiser_id_alias'] = device.name
                            
                            # Generate the report
                            report = Report(device)
                            pdf_file_path = report.file_name
                            results_c.write('Report generated!')
                            
                            # Save the report in the dictionary st.session_state.generated_reports so that if the user wants to regenerate a report on this ADID,
                            # they don't need to wait forever for the algorithms to run--it's saved already!
                            # Also helps us display in the sidebar which reports have been generated
                            st.session_state.generated_reports.loc[len(st.session_state.generated_reports)] = [adid, device.name, device]
                            
                            # Display the report in a PDF viewer within the browser
                            with open(pdf_file_path, "rb") as f:
                                pdf_bytes = f.read()
                            pdf_base64 = b64encode(pdf_bytes).decode('utf-8')
                            pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="900" height="800" type="application/pdf"></iframe>'
                            results_c.write(pdf_display, unsafe_allow_html=True)
                        
                        # If a report has been generated on this ADID already...    
                        else:
                            # Fetch the device
                            device = st.session_state.generated_reports[st.session_state.generated_reports['ADID'] == adid]['Profile'].reset_index(drop=True)[0]
                            
                            # Make the report based off of the device
                            report = Report(device)
                            pdf_file_path = report.file_name
                            results_c.write('Report generated!')
                            
                            # Display the report in a PDF viewer within the browser
                            with open(pdf_file_path, "rb") as f:
                                pdf_bytes = f.read()
                            pdf_base64 = b64encode(pdf_bytes).decode('utf-8')
                            pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="900" height="800" type="application/pdf"></iframe>'
                            results_c.write(pdf_display, unsafe_allow_html=True)
                            
                    else:
                        results_c.write('Upload Data First!')
                            
        # This block shows a sidebar with the colocated devices that were detected                
        with colocs:
            st.subheader('Colocated Devices')
            try:  
                colocs_df = st.dataframe(device.coloc.drop_duplicates()[['Alias','advertiser_id','latitude','longitude']])     
                
            except Exception as error:
                print(error)
                colocs_df = st.info('No colocated devices found')
                
        # This block shows a sidebar with the previously generated reports
        with generated_reps:
            st.subheader('Generated Reports')
            try:
                generated_report_df = st.dataframe(st.session_state.generated_reports[['Alias','ADID']])
            except:
                generated_report_df = st.info('No reports have been generated yet.')
                
# AY 25 Addition
elif nav_bar == 'Graph':

    sidebar.title('Graph')

    sidebar.write("The module below generates a graph with the given dataset.")

    sidebar.subheader('Generate Graph')

    graph_sb = sidebar.container()

    with graph_sb:
        st.title('Graph Controls')
        

        # Form for graph input controls
        graph_form = st.form(key='graph_controls_form')
        with graph_form:

            st.subheader('Initial Graph Creation')
            # Input fields for querying the graph
            adid = st.text_input('Advertiser ID')  # Placeholder for ADID query
            if adid == "":
                adid = None
            radius = st.number_input('Radius (meters)', min_value=0, value=100)
            x_time = st.number_input('Minimum Time Difference (x_time) in minutes to be together to be considered colocated', min_value=1, value=5)
            y_time = st.number_input('Minimum Time Gap (y_time) in minutes before considering repeated colocation', min_value=1, value=5)
            num_nodes = st.number_input('Maximum number of nodes to display on the graph (will display the strongest relations)', min_value=1, value=10)
            st.session_state.radius = radius
            st.session_state.x_time = x_time
            st.session_state.y_time = y_time
            st.session_state.num_nodes = num_nodes
            edge_weight_scale = 100 # st.slider(
                #'Edge Weight Scale', 
                #min_value=0, 
                #max_value=100, 
                #value=50, 
                #help="0 makes the edge completely based on frequency of colocation, while 100 makes the edge based completely on dwell time within proximity. Otherwise, it is based on a percentage of each."
            #)

            st.info(
                'Query a graph displaying the relationships of interest for a single advertising ID. '
                'The radius is the circular radius around each point for the advertising ID to search for these points of interest. '
                'The radius is in meters.'
                'For large datasets, recommend entering a specific ADID to query to speed up computation.'
                'Expect large datasets to take several minutes to first generate, and be quicker to expand.'
            )

            # Submit button for the form
            if st.form_submit_button('Generate Graph'):
                with st.spinner(text="Generating graph..."):
                    # Create the graph object
                    graph = createGraph(st.session_state.data.values.tolist(), radius)

                    # Connect related nodes in the graph with the provided parameters
                    # st.session_state.data
                    # TODO MAKE SURE THE CORRECT DATA IS INPUTTED IN HERE

                    connectNodes(graph, edge_weight_scale / 100, x_time, y_time, radius, adid, False, num_nodes)

                    # Save the graph object to session state for access in other containers
                    st.session_state.graph = graph

        graph_form_expand = st.form(key='graph_expand_form')
        with graph_form_expand:
            st.subheader('Connect Displayed ADIDs')

            st.info(
                'Use this to connect all the nodes currently displayed in the graph with eachother.'
            )
            if st.form_submit_button('Connect Current Nodes'):
                with st.spinner(text="Connecting currently displayed nodes..."):
                    # Create the graph object
                    #graph = createGraph(st.session_state.data.values.tolist(), radius)

                    # Connect related nodes in the graph with the provided parameters
                    # st.session_state.datas
                    # TODO MAKE SURE THE CORRECT DATA IS INPUTTED IN HERE

                    #connectNodes(graph, edge_weight_scale / 100, x_time, y_time, radius, adid)

                    # Save the graph object to session state for access in other containers
                    connectCurrentNodes(st.session_state.graph, x_time, radius)

        graph_form_expand2 = st.form(key='graph_expand_form2')
        with graph_form_expand2:
            st.subheader('Expand Current ADIDs')
            adid = st.text_input('Advertiser ID', value=st.session_state.get("selected_adid", ""))
            st.info(
                'Use this when querying an ADID that is currently displayed in the graph.'
                ' This will find the top neighbors connected to that ADID.'
            )
            if adid == "":
                adid = None
            if st.form_submit_button('Expand ADID'):
                with st.spinner(text="Expanding graph..."):
                    # Create the graph object
                    #graph = createGraph(st.session_state.data.values.tolist(), radius)

                    # Connect related nodes in the graph with the provided parameters
                    # st.session_state.data
                    # TODO MAKE SURE THE CORRECT DATA IS INPUTTED IN HERE
                    
                    #expandNode(st.session_state.graph, st.session_state.x_time, st.session_state.y_time, st.session_state.radius, adid, st.session_state.num_nodes)
                    expandNode(st.session_state.graph, adid, st.session_state.x_time, radius, num_nodes)
        
        graph_form_expand3 = st.form(key='graph_expand_form3')
        with graph_form_expand3:
            st.subheader('Explore New ADIDs')
            adid = st.text_input('Advertiser ID')
            st.info(
                'Use this when querying an ADID that is not currently displayed in the graph.'
                ' This will add the ADID you are querying to the graph.'
            )
            if adid == "":
                adid = None
            if st.form_submit_button('Explore ADID'):
                with st.spinner(text="Expanding graph..."):
                    # Create the graph object
                    #graph = createGraph(st.session_state.data.values.tolist(), radius)

                    # Connect related nodes in the graph with the provided parameters
                    # st.session_state.data
                    # TODO MAKE SURE THE CORRECT DATA IS INPUTTED IN HERE
                    
                    #expandNode(st.session_state.graph, st.session_state.x_time, st.session_state.y_time, st.session_state.radius, adid, st.session_state.num_nodes)
                    addADID(st.session_state.graph, adid, st.session_state.x_time, radius, num_nodes)
        
        clique_form = st.form(key='clique_query_form')
        with clique_form:
            st.subheader("Find groups for ADID")
            adid_query = st.text_input('Enter ADID to query for groups:')
            st.info(
                'Use this when querying for groups that an AdID that is apart of. '
                'A group is a subset of AdIDs where every AdID is connected to every other AdID in the set. '
                'Think of it as a group of friends where everyone is friends with everyone else within that group.'
            )
            if adid == "":
                adid = None
            submit_clique_query = st.form_submit_button("Find Groups")

        # Form for filtering edges by weight
        edge_filter_form = st.form(key='edge_weight_filter_form')
        with edge_filter_form:
            st.subheader('Filter Edges by Weight')

            # Input fields for setting weight bounds
            lower_bound = st.number_input('Lower Bound for Edge Weight', min_value=0, value=0)
            upper_bound = st.number_input('Upper Bound for Edge Weight', min_value=0, value=100)
            st.session_state.has_bounds = False
            # Button to apply the filter
            if st.form_submit_button('Apply Filter'):
                st.session_state.has_bounds = True
                with st.spinner("Filtering edges..."):
                    # Refresh the graph visualization without regenerating it
                    st.session_state.lower_bound = lower_bound
                    st.session_state.upper_bound = upper_bound

    with overview_c:
        if 'graph' in st.session_state:
            graph = st.session_state.graph
            top_adids = sorted([adid for adid in graph.top_adids if adid is not None])

            st.subheader("Top ADIDs Overview")

            for i, adid in enumerate(top_adids):
                node = graph.get_node_by_adid(adid)
                if node is None:
                    continue
                
                col1, col2, col3, col4 = st.columns([1.5, 2, 4, 4])

                with col1:
                    if st.button("Set ADID", key=f"set_adid_{i}"):
                        st.session_state.selected_adid = adid
                        st.rerun()

                with col2:
                    st.markdown(f"**{adid}**")

                with col3:
                    st.markdown("**Neighbors:**")
                    neighbor_ids = ", ".join(neighbor.adid for neighbor in node.neighbors)
                    st.markdown(neighbor_ids if neighbor_ids else "None")
    with results_c:
        if 'graph' in st.session_state:
            graph = st.session_state.graph

            # [NEW STUFF]
            # Now, generate and display the visualization
            adj_matrix = np.nan_to_num(graph.adjacency_matrix, nan=0.0)
            fig = None
            if st.session_state.has_bounds:
                fig = generate_visualization(graph, adj_matrix, graph.top_adids, min_weight=st.session_state.lower_bound, max_weight=st.session_state.upper_bound)
            else:
                fig = generate_visualization(graph, adj_matrix, graph.top_adids)  # This generates the Plotly graph and shows it directly
            st.plotly_chart(fig)  # not sure how to exactly use this...

            if submit_clique_query and adid_query:
                cliques = find_cliques_for_adid(st.session_state.graph, adid_query)
                st.subheader(f"Groups for ADID: {adid_query}")

                if cliques:
                    # Convert the cliques into a structured DataFrame
                    clique_dict = {"Group #": [], "Members": []}
                    for idx, clique in enumerate(cliques, 1):
                        clique_dict["Group #"].append(f"Group {idx}")
                        clique_dict["Members"].append(", ".join(clique))  # Keep members as a single column

                    df = pd.DataFrame(clique_dict)

                    # Display as an interactive dataframe
                    st.dataframe(df, hide_index=True) 
                else:
                    st.write(f"No groups found for ADID: {adid_query}")
            
        else:
            st.warning("Please generate a graph using the controls in the sidebar.")
else:
    pass # Nothing should happen, it should never be here


# if the button is clicked, reset the data seen by the user to what the user uploaded originally
# this is done by saving the original data to a pickle file, and reloading it
if data_reset_button:
    
    # see if the pickle file exists already
    if os.path.exists(os.path.abspath('./saved_data/saved_df.pkl')): #and os.path.exists(os.path.abspath('./saved_data/original_df.pkl')):
        # load the pickle file if it does
        try:
            pickle_path = 'saved_df' #if keep_aliases_check else 'original_df'
            
            with open(os.path.abspath(f'./saved_data/{pickle_path}.pkl'), 'rb') as pkl_file:
                with st.spinner("Reseting the data..."): 
                    # load in the pickle file and update the file source
                    st.session_state.data = pickle.load(pkl_file)    
                    #st.session_state.file_source = os.path.abspath('./saved_data/saved_df.pkl')
                    
                    # update the data overview section
                    overview_c.empty()
                    overview_c.dataframe(adid_value_counts(st.session_state.data), height=300)
                    
                    # sort and modify the columns if needed
                    st.session_state.data = modify_and_sort_columns(st.session_state.data)
   
        except:
            title_c.error('Error reseting the data. Please upload manually')
            
    # if the pickle file doesn't exist, raise an error
    else:
        title_c.error('No data has been entered yet. Please upload using the side bar on the "Data" tab')

# Preview container
with preview_c:
    # If Data means if they have uploaded a file
    if st.session_state.uploaded:
        st.dataframe(st.session_state.data)


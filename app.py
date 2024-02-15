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
from agile.utils.files import find, random_line, save, random_name
from agile.utils.dataframes import modify_and_sort_columns
from agile.profile import Profile
from agile.samsreport import Report
from agile.centrality import compute_top_centrality
from agile.overview import adid_value_counts

from streamlit_option_menu import option_menu
import pygeohash as gh

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
    for filename in os.listdir(os.path.abspath('./saved_data')):
        file_path = os.path.join(os.path.abspath('./saved_data'), filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    st.session_state.file_source = False
if 'coloc_ids' not in st.session_state:
    st.session_state.coloc_ids = pd.DataFrame(columns=['Colocated ADIDs','Alias'])
if 'generated_reports' not in st.session_state:
    st.session_state.generated_reports = pd.DataFrame(columns=['ADID', 'Alias','Profile'])


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


nav_bar = option_menu(None, ['Data', 'Filtering', 'Locations', 'Algorithms', 'Report'],
                      icons=['file-earmark-fill', 'funnel-fill', 'pin-map-fill', 'layer-forward', 'stack'],
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
keep_aliases_check = data_opts.checkbox('Keep Aliases', True)

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
        
        # If a file has not yet been uploaded (this allows multiple form requests in unison)
        if raw_data and raw_data.name != st.session_state.file_source:
            try:
                st.session_state.data = pd.read_csv(raw_data, sep=',')
                st.session_state.uploaded = True
                st.session_state.file_source = raw_data.name
                
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
                        
                    
                     
                    # Save the data to a pickle file, located in the /saved_data directory
                    # This is done so it can be reloaded with the "reset data" button
                    with st.spinner("Saving the geohashed data locally..."):
                        save('original_df.pkl',st.session_state.data)  
                        save('saved_df.pkl',st.session_state.data)   
                        
                
                # perform final preprocessing operations before displaying the data
                st.session_state.data = modify_and_sort_columns(st.session_state.data)
                    
                
                    
            except:
                results_c.error('Invalid file format. Please upload a valid .csv file that contains latitude and longitude columns.')
               
        # If there is a dataframe, update the "Data Overview" statistics 
        if st.session_state.uploaded and not data_reset_button:
            try:
                # Update the value counts for an ADID
                overview_c.dataframe(adid_value_counts(st.session_state.data), height=300)
            except:
                overview_c.error("Could not load overview statistics.")
            
            
    # Container for adding an alias to an ADID
    renamer = sidebar.container()
    with renamer:
        st.subheader('Add Alias for an ADID')
        st.write("Choose a name yourself or generate a random name for an ADID")
        
        # Creates the form which will hold the text boxes, check box, and button
        rename_form = st.form('rename_adid')
        with rename_form:
            adid_rename_text = st.text_input('Advertiser ID')
            new_name_text = st.text_input('Custom Name')
            random_name_generation = st.checkbox('Generate Random Name (will override custom name)')
                
            if st.form_submit_button('Assign Name'):
                if adid_rename_text.strip() not in st.session_state.data['advertiser_id'].values:
                    preview_c.error('Error: Invalid ADID. Please re-enter the ADID')
                elif new_name_text == '' and not random_name_generation:
                    preview_c.error('Error: Please enter at least one character for a custom name')
                elif new_name_text in st.session_state.data['advertiser_id_alias'].values and not random_name_generation:
                    preview_c.error(f'Error: The alias {new_name_text} is already in use')
                else:
                    with st.spinner('Adding Alias...'):
                        if random_name_generation:
                            new_name_text = random_name()
                        st.session_state.data.loc[st.session_state.data['advertiser_id'] == adid_rename_text, 'advertiser_id_alias'] = new_name_text
                        
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
                     The radius is in meters. To search for a specific type of location, enter the location mname into the Node field below.')
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

    sidebar.write("The module below generates a report in PDF format about a single adverter ID (a single device) in the data.\n\nThe report may take 2-5 minutes to generate.")

    sidebar.subheader('Generate Report')
    
    report_sb = sidebar.container() #'Report'
    with report_sb:
        report_c = st.container()
        colocs = st.container()
        generated_reps = st.container()
 
        with report_c:
            report_form = st.form(key='report')
            with report_form:
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

                if report_button:
                    if adid not in st.session_state.data['advertiser_id'].values:
                        results_c.error('ADID is invalid. Please enter a different ADID')
                    #elif adid_value_counts(st.session_state.data)['Occurences in Data'].get(adid) < 200:
                    elif (adid_value_counts(st.session_state.data)['Occurences in Data'].get(adid) * 1.0 / days_covered) < 200:                        
                        suff_data = 0
                    else:
                        suff_data = 1
                    if st.session_state.uploaded:
                        print(st.session_state.generated_reports['ADID'])
                        
                        if adid not in st.session_state.generated_reports['ADID']:
                            # Check if the adid has an alias added
                            if len(st.session_state.data.query('advertiser_id==@adid')['advertiser_id_alias'].unique()) > 0:
                                adid_alias = st.session_state.data.query('advertiser_id==@adid')['advertiser_id_alias'].unique()[0]
                            else:
                                adid_alias = None
                                
                            # set up the profile for the adid
                            # creating a profile also creates the LOI DataFrame, which may take a minute or two depending on the size of the data                
                            device = Profile(data=st.session_state.data, ad_id=adid,
                                            alias=adid_alias, sd = suff_data)
                            
                            
                            st.session_state.data.loc[st.session_state.data['advertiser_id'] == adid, 'advertiser_id_alias'] = device.name
                            save('saved_df.pkl',st.session_state.data)
                            
                            # generate the report
                            report = Report(device)
                            pdf_file_path = report.file_name
                            results_c.write('Report generated!')
                            
                            if adid not in st.session_state.generated_reports['ADID']:
                                st.session_state.generated_reports.loc[len(st.session_state.generated_reports)] = [adid, device.name, device]
                            
                            with open(pdf_file_path, "rb") as f:
                                pdf_bytes = f.read()

                            pdf_base64 = b64encode(pdf_bytes).decode('utf-8')
                            pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="900" height="800" type="application/pdf"></iframe>'
                            results_c.write(pdf_display, unsafe_allow_html=True)
                        else:
                            results_c.write('Upload data first!')
                            
                    else:
                        device = st.session_state.generated_reports[st.session_state.generated_reports['ADID'] == adid]['Profile'].reset_index(drop=True)[0]
                        report = Report(device)
                        pdf_file_path = report.file_name
                        results_c.write('Report generated!')
                        
                        with open(pdf_file_path, "rb") as f:
                            pdf_bytes = f.read()

                        pdf_base64 = b64encode(pdf_bytes).decode('utf-8')
                        pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="900" height="800" type="application/pdf"></iframe>'
                        results_c.write(pdf_display, unsafe_allow_html=True)
                            
                        
        with colocs:
            st.subheader('Colocated Devices')
            try:  
                colocs_df = st.dataframe(pd.DataFrame(device.coloc['advertiser_id'].unique(), columns=['Colocated ADIDs','Alias']))
                
                for adid in colocs_df['Colocated ADIDs']:
                    if len(st.session_state.data.query('advertiser_id==@adid')['advertiser_id_alias'].unique()) == 0:
                        generated_name = random_name()
                        st.session_state.data.loc[st.session_state.data['advertiser_id'] == adid, 'advertiser_id_alias'] = generated_name
                        colocs_df.loc[colocs_df['Colocated ADIDs'] == adid, 'Alias'] = generated_name
                        
                print(colocs_df)
            except:
                colocs_df = st.info('No colocated devices found')
                
        with generated_reps:
            st.subheader('Generated Reports')
            try:
                generated_report_df = st.dataframe(st.session_state.generated_reports[['Alias','ADID']])
            except:
                generated_report_df = st.info('No reports have been generated yet.')
            
else:
    pass #Nothing should happen, it should never be here


# if the button is clicked, reset the data seen by the user to what the user uploaded originally
# this is done by saving the original data to a pickle file, and reloading it
if data_reset_button:
    # replace this with the function
    
    
    # see if the pickle file exists already
    if os.path.exists(os.path.abspath('./saved_data/saved_df.pkl')) and os.path.exists(os.path.abspath('./saved_data/original_df.pkl')):
        # load the pickle file if it does
        try:
            pickle_path = 'saved_df' if keep_aliases_check else 'original_df'
            
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


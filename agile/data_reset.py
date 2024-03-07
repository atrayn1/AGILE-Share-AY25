import os
import pandas as pd
import streamlit as st
import pickle
# umport function

def reset_data(df, title_center, overview_c):
    st.session_state.uploaded = False
    # see if the pickle file exists already
    if os.path.exists('./saved_data/saved_df.pkl'):
        # load the pickle file if it does
        try:
            with open(os.path.abspath('./saved_data/saved_df.pkl'), 'rb') as pkl_file:
                with st.spinner("Reseting the data..."): 
                    st.session_state.data = pickle.load(pkl_file)    
                    st.session_state.file_source = os.path.abspath('./saved_data/saved_df.pkl')
                    st.session_state.uploaded = True
                    overview_c.dataframe(adid_value_counts(st.session_state.data), height=300)
                    
        except:
            title_center.error('Error reseting the data. Please upload manually')
    # if it doesn't, raise an error
    else:
        title_center.error('No data has been entered yet. Please upload using the side bar on the "Data" tab')

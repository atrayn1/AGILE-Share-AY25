import streamlit as st
import pandas as pd

# Final preprocessing before the dataframe is displayed. adds a new columns for aliases if neeeded, and sorts the columns of the Data Preview in a particular order
def modify_and_sort_columns(df):
    # Remove a bad column, if necessary
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis = 1)
        
    if 'advertiser_id_alias' not in df.columns:
        df['advertiser_id_alias'] = [None] * len(df)
        
    current_columns = df.columns.tolist()
    desired_columns = ['advertiser_id', 'advertiser_id_alias'] + [col for col in current_columns if col not in ['advertiser_id', 'advertiser_id_alias']]
    
    df = df[desired_columns]
    
    return df
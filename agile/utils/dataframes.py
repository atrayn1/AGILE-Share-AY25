import streamlit as st
import pandas as pd

# Clean the column names and make sure required columns are present
def clean_and_verify_columns(df):
    required = ['latitude','longitude','advertiser_id','datetime']
    data_columns = df.columns[:]
    
    for col in data_columns:
        if col.strip().lower() in required:
            required.remove(col.strip().lower())
        if 'unnamed' in col.strip().lower():
            df.drop(col, axis=1, inplace=True)
    
    if len(required) > 0:       
        raise Exception()
    
    df.set_axis([col.strip().lower() for col in df.columns], axis=1, inplace=True)
    return df

# Final preprocessing before the dataframe is displayed. adds a new columns for aliases if neeeded, and sorts the columns of the Data Preview in a particular order
def modify_and_sort_columns(df):
    # Remove a bad column, if necessary
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis = 1)
        
    if 'advertiser_id_alias' not in df.columns:
        df['advertiser_id_alias'] = [None] * len(df)
        
    # sort the columns so that the most relevant columns are to the left
    current_columns = df.columns.tolist()
    desired_columns = ['advertiser_id', 'advertiser_id_alias'] + [col for col in current_columns if col not in ['advertiser_id', 'advertiser_id_alias']]
    df = df[desired_columns]
    
    return df


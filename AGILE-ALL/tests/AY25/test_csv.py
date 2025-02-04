'''
Created using ChatGPT
'''

import pytest
import pandas as pd
from datetime import datetime

# Function to check if the CSV is valid
def is_valid_csv_format(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # Check for correct columns
        required_columns = ['advertiser_id', 'datetime', 'latitude', 'longitude']
        assert all(col in df.columns for col in required_columns), "CSV missing required columns"
        
        # Check data types
        assert df['advertiser_id'].apply(lambda x: isinstance(x, (int, str))).all(), "advertiser_id should be integer or string"
        assert pd.to_datetime(df['datetime'], errors='raise').notna().all(), "Invalid datetime format"
        assert df['latitude'].apply(lambda x: isinstance(x, (int, float)) and -90 <= x <= 90).all(), "Invalid latitude values"
        assert df['longitude'].apply(lambda x: isinstance(x, (int, float)) and -180 <= x <= 180).all(), "Invalid longitude values"
        
        return True
    except Exception as e:
        return False

# Pytest to validate CSV format
def test_valid_csv_format():
    valid_file = 'data/testing_data.csv'
    invalid_file = 'data/invalid_adid_file.csv'

    # Test with valid file
    assert is_valid_csv_format(valid_file) is True
    
    # Test with invalid file (you could create various invalid scenarios here)
    assert is_valid_csv_format(invalid_file) is False

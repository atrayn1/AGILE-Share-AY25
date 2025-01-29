import pandas as pd
import pygeohash as gh # type: ignore
from agile.utils.dataframes import modify_and_sort_columns, clean_and_verify_columns
import time

# Main script
def process_data(file_path):
    try:
        # Step 1: Read the CSV file
        data = pd.read_csv(file_path, sep=',')

        # Step 2: Clean and verify columns
        data = clean_and_verify_columns(data)

        # Step 3: Convert datetime column to datetime type
        data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
        if data['datetime'].isna().any():
            raise ValueError("Error: Could not convert some 'datetime' values to datetime format.")

        # Step 4: Check for geohash column and generate if missing or incorrect
        if 'geohash' not in data.columns or not data['geohash'].apply(lambda x: len(str(x)) == 10).all():
            data['geohash'] = data.apply(lambda d: gh.encode(d.latitude, d.longitude, precision=10), axis=1)

        # Step 5: Modify and sort columns
        data = modify_and_sort_columns(data)

        print("Data processing complete.")
        print("Processed Data:")

        return data.values.tolist(), data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

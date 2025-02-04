'''
import pytest
import pandas as pd
import numpy as np
from .testgraph import process_data
from agile.graphing import findAllFrequencyOfColocation

def expected_matrix():
    data = [
        [0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0],
        [2.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0],
        [2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0],
        [2.0, 2.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0],
        [2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 2.0, 0.0],
        [2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 2.0],
        [2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0],
    ]
    df = pd.DataFrame(data)
    return df

def test_frequencyOfColocation():
    file_path = "data/frequencyofcolocation_dataset.csv"
    df = process_data(file_path)
    colocations = findAllFrequencyOfColocation(df, 25, 5, 1500)

    print("Returned matrix:", colocations)
    print("Expected matrix:", expected_matrix())

    np.testing.assert_array_equal(
        colocations, expected_matrix(),
        err_msg="The matrix does not match the expected output."
    )
'''    
import pytest
import pandas as pd
import numpy as np
from agile.graphing import findAllFrequencyOfColocation
from .testgraph import process_data

# Prepare the dataset in DataFrame format
def create_test_dataframe():
    data = [
        {"advertiser_id": "adid_1", "datetime": "2025-01-28 12:36:12", "latitude": 40.71284517861856, "longitude": -74.00604147150814, "feature": "feature_1"},
        {"advertiser_id": "adid_2", "datetime": "2025-01-28 12:40:32", "latitude": 40.712891928351746, "longitude": -74.00611416322138, "feature": "feature_2"},
        {"advertiser_id": "adid_3", "datetime": "2025-01-28 12:44:23", "latitude": 40.71290178894406, "longitude": -74.00611638917487, "feature": "feature_3"},
        {"advertiser_id": "adid_4", "datetime": "2025-01-28 12:55:10", "latitude": 40.712763036513045, "longitude": -74.00601075553816, "feature": "feature_4"},
        {"advertiser_id": "adid_5", "datetime": "2025-01-28 12:58:43", "latitude": 40.713244081894885, "longitude": -74.00588088170429, "feature": "feature_5"},
        {"advertiser_id": "adid_6", "datetime": "2025-01-28 12:03:15", "latitude": 40.70989039722896, "longitude": -74.01438595264297, "feature": "feature_6"},
        {"advertiser_id": "adid_7", "datetime": "2025-01-28 12:10:22", "latitude": 40.72217021956463, "longitude": -74.01920444208444, "feature": "feature_7"},
        {"advertiser_id": "adid_8", "datetime": "2025-01-28 12:20:10", "latitude": 40.7175368522732, "longitude": -74.02302291522666, "feature": "feature_8"},
        {"advertiser_id": "adid_9", "datetime": "2025-01-28 12:31:05", "latitude": 40.713202532317856, "longitude": -74.02087952619804, "feature": "feature_9"},
        {"advertiser_id": "adid_10", "datetime": "2025-01-28 12:50:12", "latitude": 40.70629045762611, "longitude": -74.01984019937191, "feature": "feature_10"},
        # Repeat for the second time period
        {"advertiser_id": "adid_1", "datetime": "2025-01-28 14:36:12", "latitude": 40.71284517861856, "longitude": -74.00604147150814, "feature": "feature_1"},
        {"advertiser_id": "adid_2", "datetime": "2025-01-28 14:40:32", "latitude": 40.712891928351746, "longitude": -74.00611416322138, "feature": "feature_2"},
        {"advertiser_id": "adid_3", "datetime": "2025-01-28 14:44:23", "latitude": 40.71290178894406, "longitude": -74.00611638917487, "feature": "feature_3"},
        {"advertiser_id": "adid_4", "datetime": "2025-01-28 14:55:10", "latitude": 40.712763036513045, "longitude": -74.00601075553816, "feature": "feature_4"},
        {"advertiser_id": "adid_5", "datetime": "2025-01-28 14:58:43", "latitude": 40.713244081894885, "longitude": -74.00588088170429, "feature": "feature_5"},
        {"advertiser_id": "adid_6", "datetime": "2025-01-28 14:03:15", "latitude": 40.70989039722896, "longitude": -74.01438595264297, "feature": "feature_6"},
        {"advertiser_id": "adid_7", "datetime": "2025-01-28 14:10:22", "latitude": 40.72217021956463, "longitude": -74.01920444208444, "feature": "feature_7"},
        {"advertiser_id": "adid_8", "datetime": "2025-01-28 14:20:10", "latitude": 40.7175368522732, "longitude": -74.02302291522666, "feature": "feature_8"},
        {"advertiser_id": "adid_9", "datetime": "2025-01-28 14:31:05", "latitude": 40.713202532317856, "longitude": -74.02087952619804, "feature": "feature_9"},
        {"advertiser_id": "adid_10", "datetime": "2025-01-28 14:50:12", "latitude": 40.70629045762611, "longitude": -74.01984019937191, "feature": "feature_10"},
    ]
    return pd.DataFrame(data)

def expected_matrix():
    # Expected adjacency matrix based on test input; adjust as needed based on your logic
    data = [
        [0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0],
        [2.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0],
        [2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0],
        [2.0, 2.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0],
        [2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 2.0, 0.0],
        [2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 2.0],
        [2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0],
    ]
    return np.array(data)

def test_frequencyOfColocation():
    file_path = "data/testing_data.csv"
    data, df = process_data(file_path)
    #df = create_test_dataframe()

    # Call the function to get the colocation matrix
    colocations = findAllFrequencyOfColocation(df, x_time=25, y_time=5, radius=1500)

    # Assert the output matches the expected matrix
    np.testing.assert_array_equal(
        colocations.values, expected_matrix(),
        err_msg="The matrix does not match the expected output."
    )

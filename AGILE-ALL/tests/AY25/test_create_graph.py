import pytest
import torch
from agile.graphing import createGraph, findAllFrequencyOfColocation, dwellTimeAdjacencyMatrix, mergeResults
from .testgraph import process_data
import numpy as np

# Define the path to the CSV file
csv_file = "data/dwelltime_testset.csv"

@pytest.fixture
def setup_data():
    # Process data and create the graph
    data, df = process_data(csv_file)
    graph = createGraph(data)
    
    # Run these expensive operations only once
    frequency_result = findAllFrequencyOfColocation(df, 5, 5, 100)
    adjacency_result = dwellTimeAdjacencyMatrix(df, 5, 5, 100)
    
    return df, frequency_result, adjacency_result

def test_findAllFrequencyOfColocation(setup_data):
    df, frequency_result, _ = setup_data
    # Test for expected list of lists output for findAllFrequencyOfColocation
    expected_output = [
        [0., 1., 3., 0.],
        [1., 0., 1., 3.],
        [3., 1., 0., 0.],
        [0., 3., 0., 0.]
    ]
    
    # Use precomputed frequency result
    result = frequency_result
    
    # Check if the output matches the expected list of lists
    assert result == expected_output, f"Expected {expected_output}, but got {result}"

def test_dwellTimeAdjacencyMatrix(setup_data):
    df, _, adjacency_result = setup_data
    # Test for expected list of lists output for dwellTimeAdjacencyMatrix
    expected_output = [[0, 13.333333333333334, 34.666666666666664, 0], 
                        [13.333333333333334, 0, 13.333333333333334, 34.666666666666664], 
                        [34.666666666666664, 13.333333333333334, 0, 0], 
                        [0, 34.666666666666664, 0, 0]]
    
    
    # Use precomputed adjacency result
    result = adjacency_result
    
    # Check if the output matches the expected list of lists
    assert result == expected_output, f"Expected {expected_output}, but got {result}"

def test_mergeResults(setup_data):
    df, frequency_result, adjacency_result = setup_data
    # Test for expected list of lists output for mergeResults
    expected_output = [
        [np.nan, 13.333333015441895, 11.555556297302246, np.nan],
        [13.333333015441895, np.nan, 13.333333015441895, 11.555556297302246],
        [11.555556297302246, 13.333333015441895, np.nan, np.nan],
        [np.nan, 11.555556297302246, np.nan, np.nan]
    ]
    
    result = mergeResults(frequency_result, adjacency_result, 0)
    
    # Convert result tensor to list of lists
    result_list = result.tolist()  # Converts tensor to a Python list
    
    # Check if the output matches the expected list of lists
    assert result_list[0][1] == expected_output[0][1] and result_list[0][2] == expected_output[0][2] and result_list[1][0] == expected_output[1][0] and result_list[1][2] == expected_output[1][2] and  result_list[1][3] == expected_output[1][3] and result_list[2][0] == expected_output[2][0] and result_list[2][1] == expected_output[2][1] and result_list[3][1] == expected_output[3][1], f"Expected {expected_output}, but got {result_list}"

import numpy as np
import pytest
from agile.graphing import createGraph, connectNodes
from .testgraph import process_data

def start_visualization(csv_file, min_time_together, max_time_diff, radius):
    """
    Processes the CSV file, creates a graph, connects nodes, and generates an interactive visualization.

    Args:
        csv_file (str): Path to the CSV file.
        min_time_together (int): Minimum time spent together for connection.
        max_time_diff (int): Maximum time difference allowed between interactions.
        radius (int): The spatial radius for considering a connection.

    Returns:
        tuple: The graph object and adjacency matrix.
    """
    data, df = process_data(csv_file)
    print(df)  # Print dataframe for debugging
    graph = createGraph(data, radius)
    connectNodes(graph, 1, min_time_together, max_time_diff, radius, None)
    adj_matrix = np.nan_to_num(graph.adjacency_matrix, nan=0.0)

    #generate_visualization(graph, adj_matrix)

    return graph, adj_matrix

def test_start_visualization_hainan():
    _, adjacency_matrix = start_visualization("data/hainan.csv", 1, 15, 100)
    expected_matrix = np.array([
        [np.nan, 3.5, 3., 3., np.nan, np.nan, np.nan, np.nan, np.nan],
        [3.5, np.nan, 3., 3., np.nan, np.nan, np.nan, np.nan, np.nan],
        [3., 3., np.nan, np.nan, 8., 8., 8., np.nan, np.nan],
        [3., 3., np.nan, np.nan, np.nan, np.nan, np.nan, 6., 6.],
        [np.nan, np.nan, 8., np.nan, np.nan, 13., 13., np.nan, np.nan],
        [np.nan, np.nan, 8., np.nan, 13., np.nan, 13., np.nan, np.nan],
        [np.nan, np.nan, 8., np.nan, 13., 13., np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, 6., np.nan, np.nan, np.nan, np.nan, 6.],
        [np.nan, np.nan, np.nan, 6., np.nan, np.nan, np.nan, 6., np.nan]
    ])
    
    assert np.array_equal(np.isnan(adjacency_matrix), np.isnan(expected_matrix))
    assert np.allclose(np.nan_to_num(adjacency_matrix), np.nan_to_num(expected_matrix), atol=1e-5)

def test_start_visualization_hainan2():
    _, adjacency_matrix = start_visualization("data/hainan2.csv", 1, 15, 100)
    expected_matrix = np.array([
        [np.nan, 3.5, 3., np.nan, np.nan, np.nan],
        [3.5, np.nan, 3., np.nan, np.nan, np.nan],
        [3., 3., np.nan, 8., 8., 8.],
        [np.nan, np.nan, 8., np.nan, 13., 13.],
        [np.nan, np.nan, 8., 13., np.nan, 13.],
        [np.nan, np.nan, 8., 13., 13., np.nan]
    ])
    
    assert np.array_equal(np.isnan(adjacency_matrix), np.isnan(expected_matrix))
    assert np.allclose(np.nan_to_num(adjacency_matrix), np.nan_to_num(expected_matrix), atol=1e-5)

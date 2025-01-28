import pytest
import numpy as np
from agile.graphing import connectRelatedNodes, createGraph
from .testgraph import process_data
import pandas as pd

def expected_adjacency_matrix():
    data = [
        [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    # Create DataFrame from data
    df = pd.DataFrame(data)
    return df

@pytest.fixture
def graph_and_dataframe():
    # Process the data from the file to get a list and a dataframe
    file_path = "data/adid_dataset.csv"  # Path to the dataset file
    data, df = process_data(file_path)

    # Create the graph using the processed data
    graph = createGraph(data)

    return graph, df

def test_adjacency_matrix(graph_and_dataframe):
    graph, df = graph_and_dataframe

    # Call the function to connect related nodes
    connectRelatedNodes(graph, 100, df, 1.0)
    adjacency_matrix = graph.adjacency_matrix
    print(adjacency_matrix)
    print(type(adjacency_matrix))
    # Compare the resulting matrix to the expected matrix
    np.testing.assert_array_equal(
        adjacency_matrix, expected_adjacency_matrix(),
        err_msg="The adjacency matrix does not match the expected output."
    )

if __name__ == "__main__":
    pytest.main()

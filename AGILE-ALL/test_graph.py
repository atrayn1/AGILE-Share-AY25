import pandas as pd
from agile.graphing import createGraph, findRelatedNodesForAll
import pygeohash as gh
from agile.utils.dataframes import modify_and_sort_columns, clean_and_verify_columns


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
        print(data.head())

        return data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Path to the CSV file
csv_file = "data/adid_dataset.csv"

# Read the CSV file using pandas
data = process_data(csv_file)

# Create the graph
graph = createGraph(data)

def printNodeData():
    """
    Print details about the graph for testing.
    """
    print(f"Graph created with {graph.num_nodes} nodes.")
    print("Node data (first 5 nodes):")
    
    for i, node_features in enumerate(graph.node_features):
        if i >= 5:  # Limit output to first 5 nodes
            break
        print(f"Node {i}: {node_features}")

def print_adjacency_matrix(n=5):
    """
    Print the first n rows and columns of the adjacency matrix.

    Parameters:
        n (int): The number of rows and columns to display. Default is 5.
    """
    print("Adjacency Matrix (Partial View):")
    rows, cols = graph.adjacency_matrix.shape
    # Ensure n does not exceed matrix dimensions
    n = min(n, rows, cols)
    for i in range(n):
        print(graph.adjacency_matrix[i, :n].tolist())

def testFindRelated():
    """
    Test the function to find related nodes for all entries in a given node.
    """
    # Example: Find related nodes for the first node
    first_node_id = 0  # Adjust as needed for testing
    radius = 100  # Define a radius for neighbor search

    related_nodes = findRelatedNodesForAll(first_node_id, graph, radius)
    print(f"Related nodes for Node {first_node_id}: {related_nodes}")


printNodeData()
print_adjacency_matrix()


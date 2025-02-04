from tests.AY25.testgraph import process_data
from agile.graphing import createGraph, findAllFrequencyOfColocation, findRelatedNodes, connectRelatedNodes, frequencyOfColocation
import time

# Path to the CSV file
csv_file = "data/frequencyofcolocation_dataset.csv"

# Read the CSV file using pandas
data, df = process_data(csv_file)
print(df)
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

def print_adjacency_matrix(n=10):
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

    related_nodes = findRelatedNodes(graph.getNode(0), graph, radius, df)
    print(f"Related nodes for Node {first_node_id}: {related_nodes}")

"""
start_time = time.time()
colocations = frequencyOfColocation(df, "adid_1", "adid_2", 5, 5, 100)
end_time = time.time()
print(colocations)
elapsed_time = end_time - start_time
print(f"Execution time for frequencyOfColocation: {elapsed_time:.2f} seconds")
""" 


start_time = time.time()
colocations = findAllFrequencyOfColocation(df, 25, 5, 1500)
end_time = time.time()
print(colocations)
elapsed_time = end_time - start_time
print(f"Execution time for findAllFrequencyOfColocation: {elapsed_time:.2f} seconds")


'''
#printNodeData()
print_adjacency_matrix()
print("\n")
#testFindRelated()
start_time = time.time()
connectRelatedNodes(graph, 100, df, 1.0)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time for connectRelatedNodes: {elapsed_time:.2f} seconds")
print_adjacency_matrix()   
'''
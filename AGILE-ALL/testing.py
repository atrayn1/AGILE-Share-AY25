from tests.AY25.testgraph import process_data
from agile.graphing import createGraph, findAllFrequencyOfColocation, findRelatedNodes, connectRelatedNodes, connectNodes, frequencyOfColocation, dwellTimeAdjacencyMatrix
import time
import pandas as pd

# Path to the CSV file
csv_file = "data/4adids_dwelltime.csv"

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

def print_adjacency_matrix():
    # Convert the DataFrame to a NumPy array for easier manipulation
    adj_matrix = df.to_numpy()
    
    # Print the adjacency matrix in the desired format
    print("[")
    for row in adj_matrix:
        print(f" {list(row)},")
    print("]")


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

'''
start_time = time.time()
colocations = findAllFrequencyOfColocation(df, 25, 5, 100)
end_time = time.time()
print(colocations)
elapsed_time = end_time - start_time
print(f"Execution time for findAllFrequencyOfColocation: {elapsed_time:.2f} seconds")
'''

connectNodes(graph, 0, df, 5, 5, 100)
print(graph.adjacency_matrix.numpy())

'''
#testFindRelated()
start_time = time.time()
connectRelatedNodes(graph, 100, df, 1.0)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time for connectRelatedNodes: {elapsed_time:.2f} seconds")
print_adjacency_matrix()   
'''
'''
#testing dwell time stuff
# this is in hours btw
#print(dwellTimeWithinProximity(graph.get_nodes()[0], graph.get_nodes()[1], 100))

#print(dwellTimeWithinProximity(df, "adid_1", "adid_2"))
print(dwellTimeAdjacencyMatrix(df))
'''
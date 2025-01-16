import pandas as pd
from agile.graphing import createGraph, findRelatedNodesForAll
    
    
# Path to the CSV file
csv_file = "data/test_location.csv"

# Read the CSV file using pandas
data = pd.read_csv(csv_file)
#Ensure the CSV has the required columns: ADID, lat, lon
if not all(col in data.columns for col in ["advertiser_id", "latitude", "longitude"]):
    raise ValueError("The CSV file must contain 'advertiser_id', 'latitude', and 'longitude' columns.")

# Convert the pandas DataFrame to a list of lists for createGraph
data_list = data.values.tolist()
# Create the graph
graph = createGraph(data_list)

def printNodeData():
    # Print details about the graph for testing
    print(f"Graph created with {graph.num_nodes} nodes.")
    print("Node data (first 5 ADIDs):")
    for i, (adid, node_data) in enumerate(graph.node_features.items()):
        if i >= 5:  # Limit output to first 5 nodes
            break
        print(f"ADID: {adid}, Data: {node_data}")

def testFindRelated():
    # Assuming the graph is already created and `radius` is defined
    first_node_id = 0  # Example: First node in the graph

    # Find related nodes for all entries in the first node
    related_nodes = findRelatedNodesForAll(first_node_id, graph, 100)

    # Print the related nodes
    print(f"Related nodes for all entries in Node {first_node_id}: {related_nodes}")

testFindRelated()
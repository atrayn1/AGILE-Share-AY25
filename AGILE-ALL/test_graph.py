import pandas as pd
from agile.graphing import createGraph

# Path to the CSV file
csv_file = "data/test_location.csv"

# Read the CSV file using pandas
data = pd.read_csv(csv_file)

# Ensure the CSV has the required columns: ADID, lat, lon
if not all(col in data.columns for col in ["advertiser_id", "latitude", "longitude"]):
    raise ValueError("The CSV file must contain 'advertiser_id', 'latitude', and 'longitude' columns.")

# Convert the pandas DataFrame to a list of lists for createGraph
# Each row should be in the format: [ADID, lat, lon, additional columns...]
data_list = data.values.tolist()

# Create the graph
graph = createGraph(data_list)

# Print general graph information
print(f"Graph created with {graph.num_nodes} nodes.")
print("Adjacency matrix:")
print(graph.adjacency_matrix)

# Print the data stored within the first 5 nodes
print("\nData within the first 5 nodes:")
for node_idx in range(min(graph.num_nodes, 5)):  # Limit to the first 5 nodes
    print(f"Node {node_idx} data:")
    node_data = graph.node_features[node_idx]
    print(node_data)
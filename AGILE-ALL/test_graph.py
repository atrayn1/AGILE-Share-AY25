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
# Each row should be in the format: [ADID, lat, lon]
data_list = data[["advertiser_id", "latitude", "longitude"]].values.tolist()

# Create the graph
graph = createGraph(data_list)

# Print details about the graph for testing
print(f"Graph created with {graph.num_nodes} nodes.")
print("Node features:")
print(graph.node_features)
print("Adjacency matrix:")
print(graph.adjacency_matrix)
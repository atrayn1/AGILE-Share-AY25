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
data_list = data.values.tolist()

# Create the graph
graph = createGraph(data_list)

# Print details about the graph for testing
print(f"Graph created with {graph.num_nodes} nodes.")
print("Node data (first 5 ADIDs):")
for i, (adid, node_data) in enumerate(graph.node_features.items()):
    if i >= 5:  # Limit output to first 5 nodes
        break
    print(f"ADID: {adid}, Data: {node_data}")

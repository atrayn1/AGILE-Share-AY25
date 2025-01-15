import pandas as pd
from agile.graphing import createGraph

# Path to the CSV file
csv_file = "data/test_location.csv"

# Read the CSV file using pandas
data = pd.read_csv(csv_file)

# Ensure the CSV has the required columns: advertiser_id, latitude, longitude
required_columns = ["advertiser_id", "latitude", "longitude"]
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"The CSV file must contain the following columns: {required_columns}")

# Remove non-numeric data (keep advertiser_id and numeric columns)
data = data[required_columns]
data["latitude"] = pd.to_numeric(data["latitude"], errors="coerce")
data["longitude"] = pd.to_numeric(data["longitude"], errors="coerce")

# Drop rows with missing or invalid data
data = data.dropna()

# Convert the DataFrame to a list of lists
# Each row is in the format [advertiser_id, latitude, longitude]
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
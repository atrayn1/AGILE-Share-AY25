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

# Print the data stored in the first 5 nodes
print(f"Graph created with {graph.num_nodes} nodes.")
for i in range(min(5, graph.num_nodes)):
    print(f"Node {i} data: {graph.node_features[i]}")
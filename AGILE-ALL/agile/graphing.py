"""
Assistance from ChatGPT
January 2025
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .filtering import query_adid, query_location  # Importing the functions
import pandas as pd

class Graph:
    def __init__(self, num_nodes):
        """
        Initializes the graph.

        Args:
            num_nodes (int): Initial number of nodes in the graph.
        """
        self.num_nodes = num_nodes
        self.adjacency_matrix = torch.zeros((num_nodes, num_nodes))
        self.node_features = []  # List to store features for each node
        self.edge_features = []  # List to store edge features

    def add_node(self, features=None):
        """
        Adds a new node to the graph.

        Args:
            features (list, optional): Features of the new node. Defaults to None.
        """
        self.num_nodes += 1
        # Expand adjacency matrix
        new_adj = torch.zeros((self.num_nodes, self.num_nodes))
        new_adj[:-1, :-1] = self.adjacency_matrix
        self.adjacency_matrix = new_adj

        # Add new node features
        if features is None:
            features = []
        self.node_features.append(features)

    def remove_node(self, node):
        """
        Removes a node from the graph.

        Args:
            node (int): Index of the node to remove.
        """
        if node >= self.num_nodes or node < 0:
            raise ValueError("Node index out of bounds.")

        # Remove from adjacency matrix
        self.adjacency_matrix = torch.cat((
            self.adjacency_matrix[:node],
            self.adjacency_matrix[node + 1:]
        ), dim=0)
        self.adjacency_matrix = torch.cat((
            self.adjacency_matrix[:, :node],
            self.adjacency_matrix[:, node + 1:]
        ), dim=1)

        # Remove node features
        self.node_features.pop(node)

        self.num_nodes -= 1

    def add_edge(self, node1, node2, weight=1.0):
        """
        Adds an edge to the graph.

        Args:
            node1 (int): Index of the first node.
            node2 (int): Index of the second node.
            weight (float): Weight of the edge.
        """
        self.adjacency_matrix[node1, node2] = weight
        self.adjacency_matrix[node2, node1] = weight  # Assuming an undirected graph

    def set_node_features(self, features):
        """
        Sets the features for the nodes.

        Args:
            features (list): A list where each element is a list of features for a node.
        """
        assert len(features) == self.num_nodes, "Feature size mismatch with number of nodes."
        self.node_features = features

    def get_neighbors(self, node):
        """
        Gets the neighbors of a node.

        Args:
            node (int): Index of the node.

        Returns:
            List[int]: Indices of neighboring nodes.
        """
        neighbors = torch.where(self.adjacency_matrix[node] > 0)[0].tolist()
        return neighbors

    def forward_pass(self, node_transform=None):
        """
        Performs a forward pass on the graph.

        Args:
            node_transform (callable): A function to transform node features.

        Returns:
            List: Updated node features.
        """
        updated_features = self.node_features[:]

        for i in range(self.num_nodes):
            neighbors = self.get_neighbors(i)
            neighbor_features = [self.node_features[n] for n in neighbors]

            if node_transform:
                updated_features[i] = node_transform(self.node_features[i], neighbor_features)

        self.node_features = updated_features
        return self.node_features

def createGraph(data):
    """
    Creates a graph from the given data.
    Each node is identified by its ADID, and features include datetime, latitude, longitude,
    and any additional data starting from index 4.

    Args:
        data (list of lists): Each row contains [ADID, datetime, lat, lon, ..., additional_columns].

    Returns:
        Graph: A constructed graph with nodes and features.
    """
    graph = Graph(0)  # Start with an empty graph
    adid_to_node_map = {}  # Mapping from ADID to node index
    #df = pd.DataFrame(data)
    #print(df)
    for row in data:
        if len(row) < 4:  # Ensure there are at least 4 columns: ADID, datetime, lat, lon
            print(f"Skipping row due to insufficient columns: {row}")
            continue

        adid = row[0]
        try:
            # Extract datetime, latitude, longitude, and additional data
            datetime = row[2]
            lat, lon = float(row[3]), float(row[4])
            additional_data = row[5:]  # Treat remaining columns as individual features

            if adid not in adid_to_node_map:
                # Add a new node to the graph
                node_index = graph.num_nodes
                graph.add_node()
                adid_to_node_map[adid] = node_index

            # Update features of the node
            node_index = adid_to_node_map[adid]
            if isinstance(graph.node_features[node_index], list):
                graph.node_features[node_index].append([datetime, lat, lon] + additional_data)
            else:
                graph.node_features[node_index] = [[datetime, lat, lon] + additional_data]

        except (ValueError, IndexError) as e:
            print(f"Skipping row due to invalid data: {row} - Error: {e}")
            continue

    return graph

def findTimeTogether(adid1, adid2):
    """
    Finds the time spent together by two entities identified by their ADIDs.

    Args:
        adid1 (str): The first ADID.
        adid2 (str): The second ADID.

    Returns:
        int: The time spent together. Currently does nothing.
    """
    return 0

def findTimeAtSamePlace(adid1, adid2):
    """
    Finds the time spent at the same place by two entities identified by their ADIDs.

    Args:
        adid1 (str): The first ADID.
        adid2 (str): The second ADID.

    Returns:
        int: The time spent at the same place, not necessarily the same time. Currently does nothing.
    """
    return 0

def findWeight(adid1, adid2, timeTogether, timeAtSamePlace):
    """
    Finds the weight between two nodes identified by their ADIDs.

    Args:
        adid1 (str): The first ADID.
        adid2 (str): The second ADID.
        timeTogether (int): The amount of time spent together.
        timeAtSamePlace (int): The amount of time spent at the same place.

    Returns:
        float: The weight between the two nodes.
    """
    together_time = findTimeTogether(adid1, adid2)
    same_place_time = findTimeAtSamePlace(adid1, adid2)

    weight = 0
    if together_time >= timeTogether:
        weight += together_time
    if same_place_time >= timeAtSamePlace:
        weight += same_place_time

    return weight


def findRelatedNodesForAll(node_id: int, graph, radius: str, df: pd.DataFrame):
    """
    For a specified node, goes through each entry in the node and searches the rest of 
    the dataset for all other ADIDs that are within a specified radius using the 
    function built by the previous team: query_location()

    Parameters:
        node_id (int): The ID of the node in the graph.
        graph (Graph): The graph object created by createGraph().
        radius (str): The radius within which to search for related ads.

    Returns:
        list: A list of related ad IDs for the given node.
    """
    # Get the node's data from the graph
    node_data = graph.node_features[node_id]

    # Ensure the node has at least one entry (list) with location data (latitude and longitude)
    if len(node_data) == 0:
        raise ValueError(f"Node {node_id} does not have any feature data.")

    # Initialize the list to store related results
    all_related_results = []

    # Iterate over each entry in the node's features
    for entry in node_data:
        lat = entry[1]  # Latitude is in the second position in the list
        lon = entry[2]  # Longitude is in the third position in the list

        # Call query_location() for each entry to find related nodes within the radius
        result = query_location(lat=str(lat), long=str(lon), radius=radius, df=df)
        
        # Append the result (if not empty) to the list of all related results
        if result is not None and not result.empty:
            all_related_results.append(result)

    return all_related_results

def createEdgesFromRelatedNodes(node_id: int, graph, radius: str):
    """
    Uses the list of related nodes from findRelatedNodesForAll to create edges with a weight of 1
    between the given node and each related node.

    Parameters:
        node_id (int): The ID of the node in the graph.
        graph (Graph): The graph object created by createGraph().
        radius (str): The radius within which to search for related ads.

    Returns:
        None
    """
    # Get the related nodes for the given node
    related_nodes = findRelatedNodesForAll(node_id, graph, radius)

    # Add edges to the graph with a weight of 1
    for related_node_id in related_nodes:
        graph.add_edge(node_id, related_node_id, weight=1.0)


if __name__ == "__main__":
    createGraph()

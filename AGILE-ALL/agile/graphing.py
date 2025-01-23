"""
Assistance from ChatGPT
January 2025
MIDN 1/C Nick Summers, Alex Traynor, Anuj Sirsikar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .filtering import query_adid, query_location  # Importing the functions
import pandas as pd

class Node:
    def __init__(self, adid, features=None):
        """
        Initializes the Node.

        Args:
            adid (str): The ADID associated with this node.
            features (list, optional): Features of the node. Defaults to None.
        """
        self.adid = adid
        self.features = features if features else []
        self.neighbors = []  # List to store neighboring node indices

    def add_neighbor(self, node):
        """
        Adds a neighbor to this node.

        Args:
            node (Node): The node to be added as a neighbor.
        """
        if node not in self.neighbors:
            self.neighbors.append(node)

    def remove_neighbor(self, node):
        """
        Removes a neighbor from this node.

        Args:
            node (Node): The node to be removed from neighbors.
        """
        if node in self.neighbors:
            self.neighbors.remove(node)


class Graph:
    def __init__(self):
        """
        Initializes the graph with no nodes and an empty adjacency matrix.
        """
        self.nodes = []  # List to store nodes
        self.num_nodes = 0  # Variable to track the number of nodes
        self.adjacency_matrix = torch.zeros((0, 0))  # Initialize the adjacency matrix
    
    def get_nodes(self):
        return self.nodes

    def getNode(self, index):
        return self.nodes[index]

    def add_node(self, adid, features=None):
        """
        Adds a new node to the graph.

        Args:
            adid (str): The ADID associated with the new node.
            features (list, optional): Features of the new node. Defaults to None.
        """
        node = Node(adid, features)
        self.nodes.append(node)
        self.num_nodes += 1

        # Update the adjacency matrix to include the new node
        new_adj_matrix = torch.zeros((self.num_nodes, self.num_nodes))
        if self.num_nodes > 1:
            new_adj_matrix[:-1, :-1] = self.adjacency_matrix
        self.adjacency_matrix = new_adj_matrix

        return node

    def remove_node(self, node):
        """
        Removes a node from the graph.

        Args:
            node (Node): The node to remove from the graph.
        """
        if node in self.nodes:
            index = self.nodes.index(node)
            self.nodes.remove(node)
            self.num_nodes -= 1

            # Update adjacency matrix to reflect node removal
            self.adjacency_matrix = torch.cat((
                self.adjacency_matrix[:index],
                self.adjacency_matrix[index + 1:]
            ), dim=0)
            self.adjacency_matrix = torch.cat((
                self.adjacency_matrix[:, :index],
                self.adjacency_matrix[:, index + 1:]
            ), dim=1)

            # Also remove this node from the neighbors of all other nodes
            for n in self.nodes:
                n.remove_neighbor(node)

    def add_edge(self, node1, node2, weight):
        """
        Adds an edge between two nodes in the graph.

        Args:
            node1 (Node): The first node.
            node2 (Node): The second node.
        """
        node1.add_neighbor(node2)
        node2.add_neighbor(node1)

        # Update the adjacency matrix to reflect the edge
        idx1 = self.nodes.index(node1)
        idx2 = self.nodes.index(node2)
        self.adjacency_matrix[idx1, idx2] = weight
        self.adjacency_matrix[idx2, idx1] = weight

    def get_neighbors(self, node):
        """
        Gets the neighbors of a node.

        Args:
            node (Node): The node whose neighbors are to be retrieved.

        Returns:
            List[Node]: A list of neighboring nodes.
        """
        return node.neighbors

    def get_node_by_adid(self, adid):
        """
        Retrieves a node by its ADID.

        Args:
            adid (str): The ADID to search for.

        Returns:
            Node: The node with the specified ADID, or None if not found.
        """
        for node in self.nodes:
            if node.adid == adid:
                return node
        return None

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
    graph = Graph()  # Start with an empty graph
    adid_to_node_map = {}  # Mapping from ADID to node index

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
                node = graph.add_node(adid)  # Include ADID when adding the new node
                adid_to_node_map[adid] = node

            # Update features of the node
            node = adid_to_node_map[adid]
            node.features.append([adid, datetime, lat, lon] + additional_data)

        except (ValueError, IndexError) as e:
            print(f"Skipping row due to invalid data: {row} - Error: {e}")
            continue

    return graph

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


def findRelatedNodes(main_node, graph, radius: str, df: pd.DataFrame):
    """
    For a specified node, goes through each entry in the node and searches the rest of 
    the dataset for all other ADIDs that are within a specified radius using the 
    function built by the previous team: query_location().

    Parameters:
        main_node (Node): The node object in the graph.
        graph (Graph): The graph object created by createGraph().
        radius (str): The radius within which to search for related ads.
        df (pd.DataFrame): The dataset used for finding related nodes.

    Returns:
        pd.DataFrame: A DataFrame of related ad IDs for the given node.
    """
    # Get the node's data from the graph
    node_data = main_node.features
    exclude_adid = df.iloc[0]["advertiser_id"]
    filtered_df = df[df["advertiser_id"] != exclude_adid]

    # Ensure the node has at least one entry (list) with location data (latitude and longitude)
    if len(node_data) == 0:
        raise ValueError(f"Node {main_node.adid} does not have any feature data.")

    # Initialize the DataFrame to store related results
    all_related_results = pd.DataFrame(
        columns=["advertiser_id", "advertiser_id_alias", "datetime", "latitude", "longitude", "geohash"]
    )

    # Iterate over each entry in the node's features
    for entry in node_data:
        lat = entry[2]  # Latitude is in the second position in the list
        lon = entry[3]  # Longitude is in the third position in the list

        # Call query_location() for each entry to find related nodes within the radius
        result = query_location(lat=str(lat), long=str(lon), radius=radius, df=filtered_df)

        # If result is not empty, convert it to a DataFrame and concatenate with all_related_results
        if result is not None and len(result) > 0:
            df_result = pd.DataFrame(
                result,
                columns=["advertiser_id", "advertiser_id_alias", "datetime", "latitude", "longitude", "geohash"],
            )

            # Remove rows from filtered_df that match df_result
            filtered_df = filtered_df[~filtered_df.set_index(
                ["advertiser_id", "datetime", "latitude", "longitude", "geohash"]
            ).index.isin(
                df_result.set_index(["advertiser_id", "datetime", "latitude", "longitude", "geohash"]).index
            )]

            # Concatenate the new results into all_related_results
            all_related_results = pd.concat([all_related_results, df_result], ignore_index=True)

    return all_related_results

def connectRelatedNodesToBaseNode(base_node, graph, radius: str, df: pd.DataFrame, weight):
    """
    Runs findRelatedNodes on a specified node and creates an edge between the node
    and each unique related node (by advertiser_id) returned by findRelatedNodes.

    Parameters:
        base_node (Node): The base node to process.
        graph (Graph): The graph object.
        radius (str): The radius within which to search for related nodes.
        df (pd.DataFrame): The dataset used for finding related nodes.
        weight (float, optional): The weight of the edges to add. Defaults to 1.0.

    Returns:
        None
    """

    # Step 1: Find the related nodes based on the base_node's features
    related_nodes_adids = findRelatedNodes(base_node, graph, radius, df)
    print(related_nodes_adids)
    
    # Step 2: Iterate through each row in the DataFrame
    for _, row in related_nodes_adids.iterrows():
        adid = row["advertiser_id"]  # Extract the advertiser_id from the row
        related_node = graph.get_node_by_adid(adid)  # Get the node object using the ADID

        if related_node and related_node != base_node:
            # Step 3: Add an edge between the base node and the related node
            graph.add_edge(base_node, related_node, weight)
            print(f"Added an edge between base node {base_node.adid} and related node {related_node.adid}.")

def connectRelatedNodes(graph, radius: str, df: pd.DataFrame, weight):
    for node in graph.get_nodes():
        print(f"Running connectRelatedNodesToBaseNode for node {node.adid}.")
        connectRelatedNodesToBaseNode(node, graph, radius, df, weight)


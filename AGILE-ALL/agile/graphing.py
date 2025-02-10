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
from .prediction import haversine
import numpy as np
import math
from datetime import timedelta

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

    So bascially, this just adds one weight to nodes that share locations, but not nessecessarily 
    at the same time. So, I guess the next step would be to see if the time that the two Ad Id's 
    were at the same location, occured at the same/similiar time. Maybe that duration of time could 
    be one of the edge factors that a slider could control...

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
    """
    Connects all nodes in the graph based on proximity within a specified radius.

    This function iterates through all nodes in the graph, using the features 
    stored in each node (e.g., latitude, longitude) to identify related nodes 
    that have been within a certain radius of each other. For each pair of related nodes, 
    an edge is added between them with the specified weight.

    Parameters:
        graph (Graph): The graph object containing nodes and edges.
        radius (str): The radius within which to search for proximity connections 
                      (e.g., "10km" or "5mi").
        df (pd.DataFrame): A DataFrame containing the dataset used to check node proximity.
                           The DataFrame should include columns such as 'advertiser_id',
                           'latitude', and 'longitude' for location data.
        weight (float): The weight to assign to the edges connecting related nodes.
    """
    for node in graph.get_nodes():
        print(f"Running connectRelatedNodesToBaseNode for node {node.adid}.")
        connectRelatedNodesToBaseNode(node, graph, radius, df, weight)

def frequencyOfColocation(periods1, periods2, x_time) -> int:
    """
    Counts the number of times two ADIDs were colocated for at least x_time seconds.

    Parameters:
        periods1 (list of tuples): Continuous periods for the first ADID [(start1, end1), ...].
        periods2 (list of tuples): Continuous periods for the second ADID [(start2, end2), ...].
        x_time (int): The minimum overlap duration (in seconds) required for colocation.

    Returns:
        int: The number of times the two ADIDs were colocated for at least x_time seconds.
    """
    colocations = 0

    # Compare all pairs of continuous periods
    for start1, end1 in periods1:
        for start2, end2 in periods2:
            # Find the overlap between the two periods
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)

            # Check if they are colocated for at least x_time seconds
            overlap_duration = (overlap_end - overlap_start).total_seconds()
            if overlap_duration >= x_time:
                colocations += 1
            # Additional check: If one period fully wraps around another
            elif (start1 <= start2 and end1 >= end2) or (start2 <= start1 and end2 >= end1):
                if (end2 - start1).total_seconds() >= x_time or (end1 - start2).total_seconds() >= x_time:
                    colocations += 1

    return colocations


def findAllFrequencyOfColocation(df: pd.DataFrame, x_time: int, y_time: int, radius: float) -> list:
    """
    Computes the colocation frequency between every pair of ADIDs using precomputed continuous periods.

    Parameters:
        df (pd.DataFrame): The dataset containing ADID locations and timestamps.
        x_time (int): The maximum time difference (in minutes) to consider for the first colocation.
        y_time (int): The minimum time gap (in minutes) before considering repeated colocation.
        radius (float): The maximum distance (in meters) to consider for colocation.

    Returns:
        list: A list of lists, where each inner list contains colocation frequencies for an ADID.
    """
    # Get unique ADIDs
    unique_adids = df['advertiser_id'].unique()
    num_adids = len(unique_adids)

    # Precompute continuous periods for each ADID
    adid_periods = {adid: get_continuous_periods(df, adid, y_time*60, radius) for adid in unique_adids}

    # Initialize a list of lists to store colocation frequencies
    colocation_matrix = [[0] * num_adids for _ in range(num_adids)]

    # Iterate through all unique pairs of ADIDs
    for i in range(num_adids):
        for j in range(i + 1, num_adids):  # Avoid redundant calculations (matrix is symmetric)
            adid_1, adid_2 = unique_adids[i], unique_adids[j]
            count = frequencyOfColocation(adid_periods[adid_1], adid_periods[adid_2], x_time*60)
            colocation_matrix[i][j] = count
            colocation_matrix[j][i] = count  # Mirror the value

    return colocation_matrix


def mergeResults(adj_matrix1: list, adj_matrix2: list, x: float) -> torch.Tensor:
    """
    Merges two adjacency matrices by weighting them based on a given factor x.

    Parameters:
        adj_matrix1 (list): The first adjacency matrix (list of lists).
        adj_matrix2 (list): The second adjacency matrix (list of lists).
        x (float): A value between 0 and 1 indicating how much weight to give to adj_matrix1.
                   The remaining weight (1 - x) is given to adj_matrix2.

    Returns:
        torch.Tensor: The merged adjacency matrix as a PyTorch tensor.
    """
    if not (0 <= x <= 1):
        raise ValueError("x must be between 0 and 1")

    tensor1 = torch.tensor(adj_matrix1, dtype=torch.float32)
    tensor2 = torch.tensor(adj_matrix2, dtype=torch.float32)

    if tensor1.shape != tensor2.shape:
        raise ValueError("Adjacency matrices must have the same shape")
    print("x: ", x)
    #tensorFinal = tensor1 + tensor2
    print(tensor1)
    print(tensor2)
    tensorFinal = (tensor2) / (tensor1)

    return tensorFinal

def update_graph_with_matrix(graph, adjacency_matrix: torch.Tensor):
    """
    Updates the graph's adjacency matrix and assigns neighbors to each node.

    Parameters:
        graph: The graph object to update.
        adjacency_matrix (torch.Tensor): The new adjacency matrix.
    """
    # Update the adjacency matrix in the graph
    graph.adjacency_matrix = adjacency_matrix

    # Reset and update neighbors for each node
    num_nodes = adjacency_matrix.shape[0]
    nodes = graph.get_nodes()

    for i in range(len(nodes)):
        node = nodes[i]  # Assuming graph.nodes is index-based
        node.neighbors = []  # Reset neighbors

        for j in range(num_nodes):
            if adjacency_matrix[i, j] > 0:  # If there is a connection
                node.neighbors.append(nodes[j])

def connectNodes(graph, x, df, x_time, y_time, radius):
    """
    Connects nodes in a graph based on colocation frequency within a given time and distance.

    This function calculates the frequency of colocation between nodes (ADIDs) based on a dataset,
    merges the adjacency matrices using a weighted factor `x`, and updates the graph with the final matrix.

    Parameters:
        graph (Graph): The graph object to update.
        x (float): The weighting factor (between 0 and 1) used when merging adjacency matrices.
        df (pd.DataFrame): The dataset containing location and time data for ADIDs.
        x_time (int): The time threshold (in minutes) for considering colocation.
        y_time (int): The minimum time gap (in minutes) before considering repeated colocation.        
        radius (float): The maximum distance (in meters) for colocation.

    Returns:
        None: The function updates the graph in-place with new connections.
    """
    # Compute the adjacency matrix based on colocation frequency
    matrix1 = findAllFrequencyOfColocation(df, x_time, y_time, radius)

    # Clone the first matrix to create a second identical matrix
    matrix2 = dwellTimeAdjacencyMatrix(df, y_time*60, radius)

    # Merge the two matrices based on the weighting factor x
    finalMatrix = mergeResults(matrix1, matrix2, x)
    print(finalMatrix)
    # Update the graph with the final adjacency matrix
    update_graph_with_matrix(graph, finalMatrix)

    return matrix1, matrix2

def get_continuous_periods(df, adid, max_time_diff, max_distance):
    """
    Identify continuous time periods for a given ADID where the time between successive
    data points is less than `max_time_diff` and the distance is within `max_distance`.
    
    :param df: DataFrame containing ['advertiser_id', 'datetime', 'latitude', 'longitude']
    :param adid: The ADID to filter for
    :param max_time_diff: Maximum allowed time difference between consecutive points
    :param max_distance: Maximum allowed distance between consecutive points
    :return: List of tuples (start_time, end_time) for continuous periods
    """
    df = df[df['advertiser_id'] == adid].reset_index(drop=True)
    periods = []
    current_start = df.iloc[0]['datetime']
    current_end = df.iloc[0]['datetime']
    
    for i in range(1, len(df)):
        time_diff = (df.iloc[i]['datetime'] - df.iloc[i - 1]['datetime']).total_seconds()
        distance = haversine(df.iloc[i]['latitude'], df.iloc[i]['longitude'], 
                             df.iloc[i - 1]['latitude'], df.iloc[i - 1]['longitude'])
        
        if time_diff <= max_time_diff and distance <= max_distance:
            # Extend the current period if the time and distance conditions are met
            current_end = df.iloc[i]['datetime']
        else:
            # No overlap, record the current period and start a new one
            periods.append((current_start, current_end))
            current_start = df.iloc[i]['datetime']
            current_end = df.iloc[i]['datetime']
    
    # Add the last period
    periods.append((current_start, current_end))
    
    return periods

def dwellTimeWithinProximity(periods1, periods2):
    """
    Calculate the total overlap time between two ADIDs based on their continuous time periods.
    
    :param periods1: First ADID
    :param periods2: Second ADID
    :return: Total overlap time in seconds
    """

    print(f"1st ADID continuous periods: {periods1}")
    print(f"2nd ADID continuous periods: {periods2}")
    
    total_overlap_time = 0
    
    # Compare all pairs of periods from both ADIDs
    for start1, end1 in periods1:
        for start2, end2 in periods2:
            # Check if the periods overlap
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            
            if overlap_start < overlap_end:
                overlap_duration = (overlap_end - overlap_start).total_seconds()
                total_overlap_time += overlap_duration
                print(f"Overlap found: Start: {overlap_start}, End: {overlap_end}, Duration: {overlap_duration}s")
    
    print(f"Total overlap time: {total_overlap_time} seconds")
    return total_overlap_time

def dwellTimeAdjacencyMatrix(df, max_time_diff, max_distance):
    """
    Create an adjacency matrix of overlap times between all unique ADIDs in the dataframe.
    
    :param df: DataFrame containing ['advertiser_id', 'datetime', 'latitude', 'longitude']
    :param max_time_diff: Maximum allowed time difference between consecutive points (in seconds)
    :param max_distance: Maximum allowed distance between consecutive points (in meters)
    :return: Adjacency matrix as a list of lists, where each entry represents the overlap time between two ADIDs
    """
    # Extract unique ADIDs from the dataframe
    adids = df['advertiser_id'].unique()
    num_adids = len(adids)
    
    # Precompute continuous periods for each ADID and store them in a list
    adid_periods = [get_continuous_periods(df, adid, max_time_diff, max_distance) for adid in adids]
    
    # Initialize the adjacency matrix with zeros
    adjacency_matrix = [[0] * num_adids for _ in range(num_adids)]
    
    # Loop through each unique pair of ADIDs
    for i in range(num_adids):
        for j in range(i + 1, num_adids):  # Start from i + 1 to avoid i == j
            overlap_time = dwellTimeWithinProximity(adid_periods[i], adid_periods[j])
            adjacency_matrix[i][j] = overlap_time
            adjacency_matrix[j][i] = overlap_time  # Ensure symmetry
    
    return adjacency_matrix

# test
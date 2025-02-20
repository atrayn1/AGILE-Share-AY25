"""
Assistance from ChatGPT
January 2025
MIDN 1/C Nick Summers, Alex Traynor, Anuj Sirsikar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.edges = []  # List to store edges connected to this node
        self.continuous_periods = []  # List to store continuous time periods
        self.continuous_periods_coords = []  # List to store continuous time periods

    def getEdge(self, node):
        print("here")
        for edge in self.edges:
            print(edge.node1.adid)
            if(edge.forNode(node)):
                return edge
        return None
    
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

    def add_edge(self, edge):
        """
        Adds an edge to the list of edges connected to this node.

        Args:
            edge (Edge): The edge to be added.
        """
        self.edges.append(edge)

    def remove_edge(self, edge):
        """
        Removes an edge from the list of edges connected to this node.

        Args:
            edge (Edge): The edge to be removed.
        """
        if edge in self.edges:
            self.edges.remove(edge)

class Edge:
    def __init__(self, node1, node2, overlapping_periods=None):
        """
        Initializes an Edge between two nodes.

        Args:
            node1 (Node): The first node connected by the edge.
            node2 (Node): The second node connected by the edge.
            weight (float): The weight of the edge, such as distance or strength.
            overlapping_periods (list, optional): List of overlapping time periods between the two nodes. Defaults to None.
        """
        self.node1 = node1
        self.node2 = node2
        self.weight = 0
        self.colocation_count = 0
        self.overlap_time = 0
        self.overlap_periods = []

    def forNode(self, node):
        print(self.node1.adid)
        print(self.node2.adid)
        print(node.adid)
        if(self.node1.adid == node.adid or self.node2.adid == node.adid):
            return True
        return False
    
    def addColocationCount(self, colocation_count):
        self.colocation_count = colocation_count

    def addOverlapTime(self, overlap_time):
        self.overlap_time = overlap_time

    def addOverlapPeriods(self, overlap_periods):
        self.overlap_periods = overlap_periods
    
    def __repr__(self):
        return (f"Edge(node1={self.node1.adid}, node2={self.node2.adid}, weight={self.weight}, "
                f"colocation_count={self.colocation_count}, "
                f"overlap_time={self.overlap_time}, overlap_periods={self.overlap_periods})")

class Graph:
    """
    Represents a graph containing nodes and edges, where nodes are connected by weighted edges.

    This graph includes methods to add and remove nodes, add and remove edges, and manage the adjacency matrix.
    Additionally, nodes can store their own edges and neighbors, making graph traversal and edge-related operations more efficient.

    Attributes:
        nodes (list): A list of all nodes in the graph.
        num_nodes (int): The total number of nodes in the graph.
        edges (list): A list of all edges in the graph.
        adjacency_matrix (torch.Tensor): The adjacency matrix representing node connections.
        colocations_matrix (torch.Tensor): A matrix storing colocation data.
        dwell_time_matrix (torch.Tensor): A matrix storing dwell time data.

    Methods:
        add_node(adid, features=None): Adds a new node to the graph.
        remove_node(node): Removes a node from the graph and updates the adjacency matrix.
        add_edge(node1, node2, weight): Adds a new edge between two nodes and updates the adjacency matrix.
        remove_edge(node1, node2): Removes an edge between two nodes and updates the adjacency matrix.
        get_neighbors(node): Returns the neighbors of a given node.
        get_node_by_adid(adid): Retrieves a node by its ADID.
    """
    def __init__(self):
        """
        Initializes an empty graph with no nodes, edges, or matrices.
        """
        self.nodes = []  # List of nodes
        self.num_nodes = 0  # Number of nodes
        self.edges = []  # List of edges
        self.adjacency_matrix = torch.zeros((0, 0))  # Adjacency matrix
        self.colocations_matrix = torch.zeros((0, 0))  # Colocation matrix
        self.dwell_time_matrix = torch.zeros((0, 0))  # Dwell time matrix
        self.relationship_type = None

    def add_node(self, adid, features=None):
        """
        Adds a new node to the graph with the given ADID and optional features.

        Args:
            adid (str): The ADID of the new node.
            features (list, optional): Features associated with the node (e.g., datetime, lat, lon). Defaults to None.

        Returns:
            Node: The newly created Node object.
        """
        node = Node(adid, features)
        self.nodes.append(node)
        self.num_nodes += 1

        # Update adjacency matrix to account for the new node
        new_adj_matrix = torch.zeros((self.num_nodes, self.num_nodes))
        if self.num_nodes > 1:
            new_adj_matrix[:-1, :-1] = self.adjacency_matrix
        self.adjacency_matrix = new_adj_matrix

        return node

    def remove_node(self, node):
        """
        Removes a node from the graph and updates the adjacency matrix.

        Args:
            node (Node): The node to remove.
        """
        if node in self.nodes:
            index = self.nodes.index(node)
            self.nodes.remove(node)
            self.num_nodes -= 1

            # Update adjacency matrix
            self.adjacency_matrix = torch.cat((
                self.adjacency_matrix[:index],
                self.adjacency_matrix[index + 1:]
            ), dim=0)
            self.adjacency_matrix = torch.cat((
                self.adjacency_matrix[:, :index],
                self.adjacency_matrix[:, index + 1:]
            ), dim=1)

            # Remove this node from the neighbors of all other nodes
            for n in self.nodes:
                n.remove_neighbor(node)

    def add_edge(self, node1, node2):
        """
        Adds a new edge between two nodes and updates the adjacency matrix.

        Args:
            node1 (Node): The first node.
            node2 (Node): The second node.
        """
        edge = Edge(node1, node2)
        self.edges.append(edge)

        # Add edge to each node's list of edges
        node1.add_edge(edge)
        node2.add_edge(edge)

        # Also add neighbors
        node1.add_neighbor(node2)
        node2.add_neighbor(node1)

    def remove_edge(self, node1, node2):
        """
        Removes an edge between two nodes and updates the adjacency matrix.

        Args:
            node1 (Node): The first node.
            node2 (Node): The second node.
        """
        # Find the edge between the two nodes
        edge_to_remove = None
        for edge in self.edges:
            if (edge.node1 == node1 and edge.node2 == node2) or (edge.node1 == node2 and edge.node2 == node1):
                edge_to_remove = edge
                break

        if edge_to_remove:
            self.edges.remove(edge_to_remove)
            node1.remove_edge(edge_to_remove)
            node2.remove_edge(edge_to_remove)

            # Remove the edge from the adjacency matrix
            idx1 = self.nodes.index(node1)
            idx2 = self.nodes.index(node2)
            self.adjacency_matrix[idx1, idx2] = 0
            self.adjacency_matrix[idx2, idx1] = 0

            # Remove neighbors
            node1.remove_neighbor(node2)
            node2.remove_neighbor(node1)

    def get_neighbors(self, node):
        """
        Returns the list of neighbors for the given node.

        Args:
            node (Node): The node for which neighbors are to be returned.

        Returns:
            list: A list of neighboring nodes.
        """
        return node.neighbors

    def get_node_by_adid(self, adid):
        """
        Retrieves a node by its ADID.

        Args:
            adid (str): The ADID of the node to retrieve.

        Returns:
            Node: The node with the given ADID, or None if not found.
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

def update_graph_with_matrix(graph, adjacency_matrix: torch.Tensor, matrix1, matrix2):
    """
    Updates the graph's adjacency matrix and assigns neighbors to each node.

    Parameters:
        graph: The graph object to update.
        adjacency_matrix (torch.Tensor): The new adjacency matrix.
    """
    # Update the adjacency matrix in the graph
    graph.adjacency_matrix = adjacency_matrix
    graph.colocations_matrix = torch.tensor(matrix1, dtype=torch.float32)
    graph.dwell_time_matrix = torch.tensor(matrix2, dtype=torch.float32)

    # Reset and update neighbors for each node
    num_nodes = adjacency_matrix.shape[0]
    nodes = graph.nodes

    for i in range(len(nodes)):
        node = nodes[i]  # Assuming graph.nodes is index-based
        node.neighbors = []  # Reset neighbors

        for j in range(num_nodes):
            if adjacency_matrix[i, j] > 0:  # If there is a connection
                node.neighbors.append(nodes[j])

def connectNodes(graph, x, df, min_time_together, max_time_diff, radius):
    """
    Connects nodes in a graph based on colocation frequency within a given time and distance.

    This function calculates the frequency of colocation between nodes (ADIDs) based on a dataset,
    merges the adjacency matrices using a weighted factor `x`, and updates the graph with the final matrix.

    Parameters:
        graph (Graph): The graph object to update.
        x (float): The weighting factor (between 0 and 1) used when merging adjacency matrices.
        df (pd.DataFrame): The dataset containing location and time data for ADIDs.
        min_time_together: Minimum amount of time ADIDs must be colocated to be considered together.
        max_time_diff: Maximum allowed time difference between consecutive points (in seconds)      
        radius (float): The maximum distance (in meters) for colocation.

    Returns:
        None: The function updates the graph in-place with new connections.
    """
    findAllContinuousPeriods(graph, df, max_time_diff*60, radius)

    # Compute the adjacency matrix based on colocation frequency
    matrix1 = findAllFrequencyOfColocation(graph, min_time_together)

    # Clone the first matrix to create a second identical matrix
    matrix2 = findAllDwellTimeWithinProximity(graph, min_time_together)

    # Merge the two matrices based on the weighting factor x
    finalMatrix = mergeResults(matrix1, matrix2, x)
    print(finalMatrix)
    # Update the graph with the final adjacency matrix
    update_graph_with_matrix(graph, finalMatrix, matrix1, matrix2)

    return matrix1, matrix2

def findAllContinuousPeriods(graph, df, max_time_diff, max_distance):
    """
    Computes and sets the continuous periods for every node in the graph.

    Args:
        graph (Graph): The graph containing nodes.
        df (pd.DataFrame): The dataset to compute continuous periods.
        max_time_diff (int): Maximum allowed time difference between consecutive points.
        max_distance (float): Maximum allowed distance between consecutive points.
    """
    for node in graph.nodes:
        periods = get_continuous_periods(df, node.adid, max_time_diff, max_distance)
        node.continuous_periods = periods

def get_continuous_periods(df, adid, max_time_diff=300, max_distance=100):
    """
    Identify continuous time periods for a given ADID where the time between successive
    data points is less than `max_time_diff` and the distance is within `max_distance`.
    Also, return the average latitude and longitude for each continuous period.
    
    :param df: DataFrame containing ['advertiser_id', 'datetime', 'latitude', 'longitude']
    :param adid: The ADID to filter for
    :param max_time_diff: Maximum allowed time difference between consecutive points
    :param max_distance: Maximum allowed distance between consecutive points
    :return: Two lists: periods and coords
    """
    df = df[df['advertiser_id'] == adid].reset_index(drop=True)
    periods = []
    
    current_start = df.iloc[0]['datetime']
    current_end = df.iloc[0]['datetime']
    current_latitudes = [df.iloc[0]['latitude']]
    current_longitudes = [df.iloc[0]['longitude']]
    
    for i in range(1, len(df)):
        time_diff = (df.iloc[i]['datetime'] - df.iloc[i - 1]['datetime']).total_seconds()
        distance = haversine(df.iloc[i]['latitude'], df.iloc[i]['longitude'], 
                             df.iloc[i - 1]['latitude'], df.iloc[i - 1]['longitude'])
        
        if time_diff <= max_time_diff and distance <= max_distance:
            # Extend the current period if the time and distance conditions are met
            current_end = df.iloc[i]['datetime']
            current_latitudes.append(df.iloc[i]['latitude'])
            current_longitudes.append(df.iloc[i]['longitude'])
        else:
            # No overlap, record the current period and start a new one
            average_latitude = sum(current_latitudes) / len(current_latitudes)
            average_longitude = sum(current_longitudes) / len(current_longitudes)
            periods.append([(current_start, current_end), (average_latitude, average_longitude)])
            current_start = df.iloc[i]['datetime']
            current_end = df.iloc[i]['datetime']
            current_latitudes = [df.iloc[i]['latitude']]
            current_longitudes = [df.iloc[i]['longitude']]
    
    # Add the last period
    average_latitude = sum(current_latitudes) / len(current_latitudes)
    average_longitude = sum(current_longitudes) / len(current_longitudes)
    #print(((current_start, current_end), (average_latitude, average_longitude)))
    periods.append([(current_start, current_end), (average_latitude, average_longitude)])
    #print(periods)
    return periods

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
    for x in periods1:
        for y in periods2:
            start1, end1 = x[0]
            start2, end2 = y[0]
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

def findAllFrequencyOfColocation(graph, x_time: int) -> list:
    """
    Computes the colocation frequency between every pair of nodes using precomputed continuous periods.

    Parameters:
        graph (Graph): The graph containing nodes.
        x_time (int): The minimum overlap duration (in seconds) required for colocation.

    Returns:
        list: A list of lists, where each inner list contains colocation frequencies for a node.
    """
    num_nodes = len(graph.nodes)
    
    # Initialize a list of lists to store colocation frequencies
    colocation_matrix = [[0] * num_nodes for _ in range(num_nodes)]

    # Iterate through all unique pairs of nodes
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Avoid redundant calculations (matrix is symmetric)
            node1, node2 = graph.nodes[i], graph.nodes[j]
            count = frequencyOfColocation(node1.continuous_periods, node2.continuous_periods, x_time)
            if(count > 0):
                if(node1.getEdge(node2) == None):
                    graph.add_edge(node1, node2)
                edge = node1.getEdge(node2)
                edge.addColocationCount(count)
            colocation_matrix[i][j] = count
            colocation_matrix[j][i] = count  # Mirror the value

    return colocation_matrix

def dwellTimeWithinProximity(periods1, periods2):
    """
    Calculate the total overlap time between two ADIDs based on their continuous time periods,
    return the total overlap time in minutes and a list of overlap periods.
    
    :param periods1: First ADID's continuous periods, list of tuples (start_time, end_time)
    :param periods2: Second ADID's continuous periods, list of tuples (start_time, end_time)
    :return: Total overlap time in minutes and a list of overlap periods as tuples (start_time, end_time)
    """

    #print(f"1st ADID continuous periods: {periods1}")
    #print(f"2nd ADID continuous periods: {periods2}")
    
    total_overlap_time = 0
    overlap_periods = []
    
    # Compare all pairs of periods from both ADIDs
    for x in periods1:
        for y in periods2:
            start1, end1 = x[0]
            start2, end2 = y[0]
            # Check if the periods overlap
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            
            if overlap_start < overlap_end:
                overlap_duration = (overlap_end - overlap_start).total_seconds()
                total_overlap_time += overlap_duration
                overlap_periods.append([(overlap_start, overlap_end), x[1]])
                #print(f"Overlap found: Start: {overlap_start}, End: {overlap_end}, Duration: {overlap_duration}s")

    
    #print(f"Total overlap time: {total_overlap_time} seconds")
    
    # Return total overlap time in minutes and the list of overlap periods
    return total_overlap_time / 60, overlap_periods

def findAllDwellTimeWithinProximity(graph, min_time_together):
    """
    Create an adjacency matrix of overlap times between all nodes in the graph.
    
    graph: Graph containing nodes with precomputed continuous periods.
    min_time_together: Minimum amount of time nodes must be colocated to be considered together (in minutes).

    :return: Adjacency matrix as a list of lists, where each entry represents the overlap time between two nodes.
    """
    num_nodes = len(graph.nodes)
    
    # Initialize the adjacency matrix with zeros
    adjacency_matrix = [[0] * num_nodes for _ in range(num_nodes)]

    adjacency_matrix_periods = [[0] * num_nodes for _ in range(num_nodes)]
    
    # Loop through each unique pair of nodes
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Start from i + 1 to avoid i == j
            node1, node2 = graph.nodes[i], graph.nodes[j]
            overlap_time, overlap_periods = dwellTimeWithinProximity(node1.continuous_periods, node2.continuous_periods)
            if(overlap_time > 0):
                if(node1.getEdge(node2) == None):
                    graph.add_edge(node1, node2)
                edge = node1.getEdge(node2)
                edge.addOverlapTime(overlap_time)
                edge.addOverlapPeriods(overlap_periods)

            print("overlap, ", overlap_periods)
            if overlap_time < min_time_together:
                overlap_time = 0
            adjacency_matrix[i][j] = overlap_time
            adjacency_matrix[j][i] = overlap_time  # Ensure symmetry
            
    return adjacency_matrix

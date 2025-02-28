"""
Assistance from ChatGPT
January 2025
MIDN 1/C Nick Summers, Alex Traynor, Anuj Sirsikar
"""

import pandas as pd
from .prediction import haversine
import numpy as np
import math
from datetime import timedelta
import sys

class Node:
    def __init__(self, adid, features=None):
        """
        Initializes the Node.

        Args:
            adid (str): The ADID associated with this node.
            features (list, optional): Features of the node. Defaults to None.
        """
        self.adid = adid
        self.original_datapoints = [] # List to store all lats/longs/timestamps
        self.neighbors = []  # List to store neighboring node indices
        self.edges = []  # List to store edges connected to this node
        self.continuous_periods = []  # List to store continuous time periods
        self.squares = [] # List to store what square something is a part of

    def getEdge(self, node):
        #print("here")
        for edge in self.edges:
            #print(edge.node1.adid)
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
        if(self.node1.adid == node.adid or self.node2.adid == node.adid):
            return True
        return False
    
    def addColocationCount(self, colocation_count):
        self.colocation_count = colocation_count

    def addOverlapTime(self, overlap_time):
        self.overlap_time = overlap_time

    def addOverlapPeriods(self, overlap_periods):
        self.overlap_periods = overlap_periods
    
    def fixWeight(self):
        if self.colocation_count is not 0:
            self.weight = self.overlap_time / self.colocation_count

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
        adjacency_matrix (np.ndarray): The adjacency matrix representing node connections.
        colocations_matrix (np.ndarray): A matrix storing colocation data.
        dwell_time_matrix (np.ndarray): A matrix storing dwell time data.

    Methods:
        add_node(adid, features=None): Adds a new node to the graph.
        remove_node(node): Removes a node from the graph and updates the adjacency matrix.
        add_edge(node1, node2): Adds a new edge between two nodes and updates the adjacency matrix.
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
        self.adjacency_matrix = np.zeros((0, 0))  # Adjacency matrix
        self.colocations_matrix = np.zeros((0, 0))  # Colocation matrix
        self.dwell_time_matrix = np.zeros((0, 0))  # Dwell time matrix
        self.relationship_type = None
        self.grid = None

    def get_edge(self, node1, node2):
        for edge in self.edges:
            if edge.node1 is node1 and edge.node2 is node2:
                return edge
        return None
    
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
        new_adj_matrix = np.zeros((self.num_nodes, self.num_nodes))
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

            # Update adjacency matrix by removing the corresponding row and column
            self.adjacency_matrix = np.concatenate((
                self.adjacency_matrix[:index, :],
                self.adjacency_matrix[index + 1:, :]
            ), axis=0)
            self.adjacency_matrix = np.concatenate((
                self.adjacency_matrix[:, :index],
                self.adjacency_matrix[:, index + 1:]
            ), axis=1)

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

        return edge

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
    
    def get_node_names(self):
        """
        Returns a list of ADID values for all nodes in the graph.
        
        Returns:
            list: A list containing the adid of each node.
        """
        return [node.adid for node in self.nodes]
    
def createGraph(data, radius):
    """
    Creates a graph from the given data.
    Each node is identified by its ADID, and features include datetime, latitude, longitude,
    and any additional data starting from index 4.

    Args:
        data (list of lists): Each row contains [ADID, datetime, lat, lon, ..., additional_columns].

    Returns:
        Graph: A constructed graph with nodes and features.
    """
    grid, min_lat, max_lat, min_lon, max_lon = preprocess(data, radius)
    
    graph = Graph()
    graph.grid = grid
    adid_to_node_map = {}

    for row in data:
        if len(row) < 4:
            print(f"Skipping row due to insufficient columns: {row}")
            continue

        adid = row[0]
        try:
            debug_print("Creating graph: Adding {} node.".format(adid))
            datetime = row[2]
            coords = (float(row[3]), float(row[4]))
            additional_data = row[5:]

            if adid not in adid_to_node_map:
                node = graph.add_node(adid)
                adid_to_node_map[adid] = node

            node = adid_to_node_map[adid]
            node.original_datapoints.append([datetime, coords])
            
            # Assign grid square index to the node
            #print(coords)
            #print((min_lat, min_lon))
            #print((max_lat, max_lon))
            rows, cols = get_grid_square(coords, (min_lat, min_lon), (max_lat, max_lon), width_meters=100)
            node.squares.append((rows, cols))
            graph.grid[rows, cols].append(len(graph.nodes)-1)
        
        except (ValueError, IndexError) as e:
            print(f"Skipping row due to invalid data: {row} - Error: {e}")
            continue
    
    return graph

def mergeResults(adj_matrix1: list, adj_matrix2: list, x: float) -> np.ndarray:
    """
    Merges two adjacency matrices by performing element-wise division of the second matrix by the first matrix.

    Parameters:
        adj_matrix1 (list): The first adjacency matrix (list of lists).
        adj_matrix2 (list): The second adjacency matrix (list of lists).
        x (float): Unused parameter, retained for compatibility.

    Returns:
        np.ndarray: The merged adjacency matrix as a NumPy array.
    """
    debug_print("Merging matricies...")
    array1 = np.array(adj_matrix1, dtype=np.float32)
    array2 = np.array(adj_matrix2, dtype=np.float32)

    if array1.shape != array2.shape:
        raise ValueError("Adjacency matrices must have the same shape")
    
    return array2 / array1

def update_graph_with_matrix(graph, adjacency_matrix: np.ndarray, matrix1, matrix2):
    """
    Updates the graph's adjacency matrix and assigns neighbors to each node.

    Parameters:
        graph: The graph object to update.
        adjacency_matrix (np.ndarray): The new adjacency matrix.
        matrix1 (list or np.ndarray): The first matrix (e.g., colocation data).
        matrix2 (list or np.ndarray): The second matrix (e.g., dwell time data).
    """
    # Update the adjacency matrix in the graph
    #debug_print("Updating graph with matrix...")
    graph.adjacency_matrix = np.array(adjacency_matrix, dtype=np.float32)
    graph.colocations_matrix = np.array(matrix1, dtype=np.float32)
    graph.dwell_time_matrix = np.array(matrix2, dtype=np.float32)

    # Reset and update neighbors for each node
    num_nodes = adjacency_matrix.shape[0]
    nodes = graph.nodes

    for i in range(len(nodes)):
        node = nodes[i]  # Assuming graph.nodes is index-based
        node.neighbors = []  # Reset neighbors

        for j in range(num_nodes):
            if adjacency_matrix[i, j] > 0:  # If there is a connection
                node.neighbors.append(nodes[j])

def connectNodes(graph, x, min_time_together, max_time_diff, radius):
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
    findAllContinuousPeriods(graph, max_time_diff*60, radius)

    matrix1, matrix2 = findAllFrequencyAndDwellTime(graph, min_time_together, radius)
    """
    # Compute the adjacency matrix based on colocation frequency
    matrix1 = findAllFrequencyOfColocation(graph, min_time_together, radius)

    # Clone the first matrix to create a second identical matrix
    matrix2 = findAllDwellTimeWithinProximity(graph, min_time_together, radius)
    """

    # Merge the two matrices based on the weighting factor x
    finalMatrix = mergeResults(matrix1, matrix2, x)
    print(finalMatrix)
    # Update the graph with the final adjacency matrix
    update_graph_with_matrix(graph, finalMatrix, matrix1, matrix2)

    return matrix1, matrix2
    
def findAllContinuousPeriods(graph, max_time_diff, max_distance):
    """
    Computes and sets the continuous periods for every node in the graph.

    Args:
        graph (Graph): The graph containing nodes.
        df (pd.DataFrame): The dataset to compute continuous periods.
        max_time_diff (int): Maximum allowed time difference between consecutive points.
        max_distance (float): Maximum allowed distance between consecutive points.
    """
    debug_print("Finding continuous periods...")
    for node in graph.nodes:
        periods = get_continuous_periods(node, max_time_diff, max_distance)
        node.continuous_periods = periods

def get_continuous_periods(node, max_time_diff, max_distance):
    """
    Identify continuous time periods for a given node where the time between successive
    data points is less than `max_time_diff` and the distance is within `max_distance`.
    Also, return the average latitude and longitude for each continuous period.
    
    :param node: Node object containing `original_datapoints` as a list of [timestamp, (lat, long)]
    :param max_time_diff: Maximum allowed time difference between consecutive points (in seconds)
    :param max_distance: Maximum allowed distance between consecutive points
    :return: List of tuples containing continuous periods and their average coordinates
    """
    datapoints = node.original_datapoints
    if not datapoints:
        return []
    
    periods = []
    current_start = datapoints[0][0]
    current_end = datapoints[0][0]
    current_latitudes = [datapoints[0][1][0]]
    current_longitudes = [datapoints[0][1][1]]
    
    for i in range(1, len(datapoints)):
        time_diff = (datapoints[i][0] - datapoints[i - 1][0]).total_seconds()
        lat1, long1 = datapoints[i][1]
        lat2, long2 = datapoints[i-1][1]
        distance = haversine(lat1, long1, lat2, long2)
        
        if time_diff <= max_time_diff and distance <= max_distance:
            # Extend the current period if the time and distance conditions are met
            current_end = datapoints[i][0]
            current_latitudes.append(datapoints[i][1][0])
            current_longitudes.append(datapoints[i][1][1])
        else:
            # No overlap, record the current period and start a new one
            average_latitude = sum(current_latitudes) / len(current_latitudes)
            average_longitude = sum(current_longitudes) / len(current_longitudes)
            periods.append([(current_start, current_end), (average_latitude, average_longitude)])
            
            current_start = datapoints[i][0]
            current_end = datapoints[i][0]
            current_latitudes = [datapoints[i][1][0]]
            current_longitudes = [datapoints[i][1][1]]
    
    # Add the last period
    average_latitude = sum(current_latitudes) / len(current_latitudes)
    average_longitude = sum(current_longitudes) / len(current_longitudes)
    periods.append([(current_start, current_end), (average_latitude, average_longitude)])
    
    return periods

def frequencyOfColocation(periods1, periods2, x_time, radius) -> int:
    """
    Counts the number of times two ADIDs were colocated for at least x_time seconds.

    This function uses a two-pointer approach to efficiently find overlapping periods
    between two lists of continuous time periods. Instead of comparing every pair 
    (which would be O(N × M)), it processes the periods in a single pass (O(N + M)).

    How It Works:
    - Both lists `periods1` and `periods2` are assumed to be sorted by start time.
    - Two pointers (`i` for `periods1`, `j` for `periods2`) iterate through both lists.
    - If a period from one list ends before the other starts, move to the next period.
    - If there is an overlap, calculate the overlap duration.
    - If the overlap duration meets or exceeds `x_time`, count it as a colocation.
    - Always advance the pointer for the period that ends first to avoid unnecessary comparisons.

    Efficiency:
    - **O(N + M) Time Complexity**: Each period is processed only once instead of checking all pairs.
    - **Short-Circuiting**: If periods don’t overlap, the function skips unnecessary calculations.
    - **Memory Efficient**: It only uses two integer counters (`i` and `j`), making it memory-friendly.

    Parameters:
        periods1 (list of tuples): Continuous periods for the first ADID [(start1, end1), ...].
        periods2 (list of tuples): Continuous periods for the second ADID [(start2, end2), ...].
        x_time (int): The minimum overlap duration (in seconds) required for colocation.

    Returns:
        int: The number of times the two ADIDs were colocated for at least x_time seconds.
    """
    colocations = 0
    i, j = 0, 0
    
    while i < len(periods1) and j < len(periods2):
        start1, end1 = periods1[i][0]
        start2, end2 = periods2[j][0]

        # If one period ends before the other starts, move to the next relevant period
        if end1 < start2:
            i += 1
            continue
        if end2 < start1:
            j += 1
            continue
        
        lat1, long1 = periods1[i][1]
        lat2, long2 = periods2[j][1]
        
        distance = haversine(lat1, long1, lat2, long2)

        # Compute overlap duration only if there is a valid overlap
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        overlap_duration = (overlap_end - overlap_start).total_seconds()

        if overlap_duration >= x_time and distance <= radius:
            colocations += 1

        # Move the pointer for the period that ends first to continue processing
        if end1 < end2:
            i += 1
        else:
            j += 1
    
    return colocations

def findAllFrequencyOfColocation(graph, x_time: int, radius: float):
    """
    Determines the frequency of colocation between all unique pairs of nodes in the graph,
    taking into account both temporal overlap and spatial proximity. Spatial proximity is
    now determined by checking the grid squares in which nodes are located and the prebuilt
    grid_lookup for adjacent cells.

    Parameters:
        graph (Graph): The graph containing nodes.
        x_time (int): The minimum overlap duration (in seconds) required for colocation.
        radius (float): The maximum distance (in meters) within which nodes should be considered.
    
    Returns:
        list: A colocation matrix (2D list) with frequencies.
    """
    debug_print("Finding frequency of colocations...")
    num_nodes = len(graph.nodes)
    colocation_matrix = [[0] * num_nodes for _ in range(num_nodes)]
    
    # Get the grid boundaries from graph.grid
    rows = [key[0] for key in graph.grid.keys()]
    cols = [key[1] for key in graph.grid.keys()]
    
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    
    # Iterate through all unique pairs of nodes.
    for i in range(num_nodes):
        node1 = graph.nodes[i]
        
        # Initialize adjacent nodes set.
        adjacent_nodes = set()

        # For each square in node1.squares, add adjacent squares and the node itself if valid.
        for row, col in node1.squares:
            # Check if the square (row, col) is within bounds.
            if min_row <= row <= max_row and min_col <= col <= max_col:
                if (row, col) in graph.grid:
                    # Add the nodes from this square to adjacent_nodes.
                    for adj_node in graph.grid[(row, col)]:
                        #print(adj_node)
                        #print(graph.nodes[1])
                        adjacent_nodes.add(graph.nodes[adj_node])
            
            # Check and add the nodes from the adjacent squares (top, bottom, left, right, diagonals).
            # Top
            if row - 1 >= min_row:
                if (row - 1, col) in graph.grid:
                    for adj_node in graph.grid[(row - 1, col)]:
                        adjacent_nodes.add(graph.nodes[adj_node])
            # Bottom
            if row + 1 <= max_row:
                if (row + 1, col) in graph.grid:
                    for adj_node in graph.grid[(row + 1, col)]:
                        adjacent_nodes.add(graph.nodes[adj_node])
            # Left
            if col - 1 >= min_col:
                if (row, col - 1) in graph.grid:
                    for adj_node in graph.grid[(row, col - 1)]:
                        adjacent_nodes.add(graph.nodes[adj_node])
            # Right
            if col + 1 <= max_col:
                if (row, col + 1) in graph.grid:
                    for adj_node in graph.grid[(row, col + 1)]:
                        adjacent_nodes.add(graph.nodes[adj_node])
            # Top-left diagonal
            if row - 1 >= min_row and col - 1 >= min_col:
                if (row - 1, col - 1) in graph.grid:
                    for adj_node in graph.grid[(row - 1, col - 1)]:
                        adjacent_nodes.add(graph.nodes[adj_node])
            # Top-right diagonal
            if row - 1 >= min_row and col + 1 <= max_col:
                if (row - 1, col + 1) in graph.grid:
                    for adj_node in graph.grid[(row - 1, col + 1)]:
                        adjacent_nodes.add(graph.nodes[adj_node])
            # Bottom-left diagonal
            if row + 1 <= max_row and col - 1 >= min_col:
                if (row + 1, col - 1) in graph.grid:
                    for adj_node in graph.grid[(row + 1, col - 1)]:
                        adjacent_nodes.add(graph.nodes[adj_node])
            # Bottom-right diagonal
            if row + 1 <= max_row and col + 1 <= max_col:
                if (row + 1, col + 1) in graph.grid:
                    for adj_node in graph.grid[(row + 1, col + 1)]:
                        adjacent_nodes.add(graph.nodes[adj_node])

        # Now, iterate through the collected adjacent nodes.
        for node2 in adjacent_nodes:
            # Skip comparing the node to itself.
            if node1 == node2:
                continue

            debug_print(f"Comparing node {node1.adid} ({i}) to {node2.adid} out of {num_nodes} nodes total, {len(adjacent_nodes)} adjacent nodes to check.")
            
            # Check for possible temporal overlap.
            if node1.continuous_periods[-1][0][1] < node2.continuous_periods[0][0][0] or \
               node2.continuous_periods[-1][0][1] < node1.continuous_periods[0][0][0]:
                continue  # No overlap in time.
            
            # Calculate temporal colocation frequency.
            count = frequencyOfColocation(node1.continuous_periods, node2.continuous_periods, x_time, radius)
            if count > 0:
                edge = node1.getEdge(node2) or graph.add_edge(node1, node2)
                edge.addColocationCount(count)
            
            # Fill in the colocation matrix symmetrically.
            colocation_matrix[i][graph.nodes.index(node2)] = count
            colocation_matrix[graph.nodes.index(node2)][i] = count  # No need for redundant computation.

    return colocation_matrix

def dwellTimeWithinProximity(periods1, periods2, radius):
    """
    Calculate the total overlap time between two ADIDs based on their continuous time periods,
    ensuring they are within the specified proximity radius.
    
    :param periods1: First ADID's continuous periods, list of tuples (start_time, end_time, (lat, lon))
    :param periods2: Second ADID's continuous periods, list of tuples (start_time, end_time, (lat, lon))
    :param radius: Maximum distance (in meters) within which nodes should be considered.
    
    :return: Total overlap time in minutes and a list of overlap periods.
    """    
    total_overlap_time = 0
    overlap_periods = []
    i, j = 0, 0
    
    while i < len(periods1) and j < len(periods2):
        time1, loc1 = periods1[i]
        time2, loc2 = periods2[j]

        start1, end1 = time1
        start2, end2 = time2
        lat1, long1 = loc1
        lat2, long2 = loc2

        # If one period ends before the other starts, move to the next relevant period
        
        if end1 < start2:
            i += 1
            continue
        if end2 < start1:
            j += 1
            continue
        
        # Compute overlap duration
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        overlap_duration = (overlap_end - overlap_start).total_seconds()

        # Check proximity before adding overlap
        if overlap_duration > 0:
            distance = haversine(lat1, long1, lat2, long2)
            if distance <= radius:
                total_overlap_time += overlap_duration
                overlap_periods.append([(overlap_start, overlap_end), (lat1, long1)])

        # Move the pointer for the period that ends first
        if end1 < end2:
            i += 1
        else:
            j += 1

    return total_overlap_time / 60, overlap_periods  # Return minutes

def findAllDwellTimeWithinProximity(graph, min_time_together, radius):
    """
    Create an adjacency matrix of overlap times between all nodes in the graph,
    considering both temporal and spatial proximity.
    
    :param graph: Graph containing nodes with precomputed continuous periods.
    :param min_time_together: Minimum time (in minutes) required for colocation.
    :param radius: Maximum distance (in meters) for nodes to be considered colocated.

    :return: Adjacency matrix as a list of lists, where each entry represents the dwell time between two nodes.
    """
    debug_print("Finding dwell time within proximity...")

    num_nodes = len(graph.nodes)
    adjacency_matrix = [[0] * num_nodes for _ in range(num_nodes)]

    # Get the grid boundaries from graph.grid
    rows = [key[0] for key in graph.grid.keys()]
    cols = [key[1] for key in graph.grid.keys()]
    
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    for i in range(num_nodes):
        node1 = graph.nodes[i]
        adjacent_nodes = set()

        # For each square in node1.squares, add adjacent squares and the node itself if valid.
        for row, col in node1.squares:
            # Check if the square (row, col) is within bounds.
            if min_row <= row <= max_row and min_col <= col <= max_col:
                if (row, col) in graph.grid:
                    # Add the nodes from this square to adjacent_nodes.
                    for adj_node in graph.grid[(row, col)]:
                        #print(adj_node)
                        #print(graph.nodes[1])
                        adjacent_nodes.add(graph.nodes[adj_node])
            
            # Check and add the nodes from the adjacent squares (top, bottom, left, right, diagonals).
            # Top
            if row - 1 >= min_row:
                if (row - 1, col) in graph.grid:
                    for adj_node in graph.grid[(row - 1, col)]:
                        adjacent_nodes.add(graph.nodes[adj_node])
            # Bottom
            if row + 1 <= max_row:
                if (row + 1, col) in graph.grid:
                    for adj_node in graph.grid[(row + 1, col)]:
                        adjacent_nodes.add(graph.nodes[adj_node])
            # Left
            if col - 1 >= min_col:
                if (row, col - 1) in graph.grid:
                    for adj_node in graph.grid[(row, col - 1)]:
                        adjacent_nodes.add(graph.nodes[adj_node])
            # Right
            if col + 1 <= max_col:
                if (row, col + 1) in graph.grid:
                    for adj_node in graph.grid[(row, col + 1)]:
                        adjacent_nodes.add(graph.nodes[adj_node])
            # Top-left diagonal
            if row - 1 >= min_row and col - 1 >= min_col:
                if (row - 1, col - 1) in graph.grid:
                    for adj_node in graph.grid[(row - 1, col - 1)]:
                        adjacent_nodes.add(graph.nodes[adj_node])
            # Top-right diagonal
            if row - 1 >= min_row and col + 1 <= max_col:
                if (row - 1, col + 1) in graph.grid:
                    for adj_node in graph.grid[(row - 1, col + 1)]:
                        adjacent_nodes.add(graph.nodes[adj_node])
            # Bottom-left diagonal
            if row + 1 <= max_row and col - 1 >= min_col:
                if (row + 1, col - 1) in graph.grid:
                    for adj_node in graph.grid[(row + 1, col - 1)]:
                        adjacent_nodes.add(graph.nodes[adj_node])
            # Bottom-right diagonal
            if row + 1 <= max_row and col + 1 <= max_col:
                if (row + 1, col + 1) in graph.grid:
                    for adj_node in graph.grid[(row + 1, col + 1)]:
                        adjacent_nodes.add(graph.nodes[adj_node])


        # Now, iterate through adjacent nodes only
        for node2 in adjacent_nodes:
            if node1 == node2:
                continue  # Skip self comparison
            
            overlap_time, overlap_periods = dwellTimeWithinProximity(
                node1.continuous_periods, node2.continuous_periods, radius
            )

            if overlap_time > 0:
                edge = node1.getEdge(node2) or graph.add_edge(node1, node2)
                edge.addOverlapTime(overlap_time)
                edge.addOverlapPeriods(overlap_periods)

            # Apply minimum time threshold
            if overlap_time < min_time_together:
                overlap_time = 0

            adjacency_matrix[i][graph.nodes.index(node2)] = overlap_time
            adjacency_matrix[graph.nodes.index(node2)][i] = overlap_time  # Ensure symmetry

    return adjacency_matrix

def findFrequencyAndDwellTime(periods1, periods2, min_time_together, radius):
    """
    This function calculates two things:
    1. The frequency of colocations where two ADIDs were colocated for at least `x_time` seconds.
    2. The total overlap time between the two ADIDs based on their continuous time periods,
       ensuring they are within the specified proximity radius.

    Parameters:
        periods1 (list of tuples): Continuous periods for the first ADID [(start1, end1), ...].
        periods2 (list of tuples): Continuous periods for the second ADID [(start2, end2), ...].
        x_time (int): The minimum overlap duration (in seconds) required for colocation.
        radius (float): Maximum distance (in meters) within which nodes should be considered.

    Returns:
        tuple: A tuple containing three values:
            - colocations (int): The number of times the two ADIDs were colocated for at least `x_time` seconds.
            - total_overlap_time (float): The total overlap time in minutes.
            - overlap_periods (list): A list of overlap periods in the format [(overlap_start, overlap_end), (lat, lon)].
    """
    colocations = 0
    total_overlap_time = 0
    overlap_periods = []
    i, j = 0, 0
    
    while i < len(periods1) and j < len(periods2):
        time1, loc1 = periods1[i]
        time2, loc2 = periods2[j]

        start1, end1 = time1
        start2, end2 = time2
        lat1, long1 = loc1
        lat2, long2 = loc2

        # If one period ends before the other starts, move to the next relevant period
        if end1 < start2:
            i += 1
            continue
        if end2 < start1:
            j += 1
            continue
        
        # Compute overlap duration
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        overlap_duration = (overlap_end - overlap_start).total_seconds()

        # Check proximity before adding overlap
        if overlap_duration > 0:
            distance = haversine(lat1, long1, lat2, long2)
            
            # For frequency of colocation
            if overlap_duration >= min_time_together and distance <= radius:
                colocations += 1
            
            # For dwell time within proximity
            if distance <= radius:
                total_overlap_time += overlap_duration
                overlap_periods.append([(overlap_start, overlap_end), (lat1, long1)])

        # Move the pointer for the period that ends first
        if end1 < end2:
            i += 1
        else:
            j += 1

    return colocations, total_overlap_time / 60, overlap_periods  # Return overlap time in minutes

def findAllFrequencyAndDwellTime(graph, min_time_together, radius):
    """
    Create two adjacency matrices:
    1. One for dwell times between all nodes considering both temporal and spatial proximity.
    2. One for the frequency of colocations between all unique pairs of nodes based on temporal overlap and spatial proximity.

    Parameters:
        graph (Graph): The graph containing nodes.
        min_time_together (int): The minimum time (in minutes) required for colocation.
        radius (float): The maximum distance (in meters) within which nodes should be considered.

    Returns:
        tuple: Two adjacency matrices as lists of lists:
            - The first matrix is for dwell times.
            - The second matrix is for colocation frequencies.
    """
    debug_print("Finding frequency of colocations and dwell time within proximity...")

    num_nodes = len(graph.nodes)
    dwell_time_matrix = [[0] * num_nodes for _ in range(num_nodes)]
    colocation_matrix = [[0] * num_nodes for _ in range(num_nodes)]
    
    # Get the grid boundaries from graph.grid
    rows = [key[0] for key in graph.grid.keys()]
    cols = [key[1] for key in graph.grid.keys()]
    dwell_time_matrix
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    # Iterate through all nodes in the graph
    for i in range(num_nodes):
        

        node1 = graph.nodes[i]
        adjacent_nodes = set()

        # For each square in node1.squares, add adjacent squares and the node itself if valid.
        for row, col in node1.squares:
            # Check if the square (row, col) is within bounds.
            if min_row <= row <= max_row and min_col <= col <= max_col:
                if (row, col) in graph.grid:
                    # Add the nodes from this square to adjacent_nodes.
                    for adj_node in graph.grid[(row, col)]:
                        adjacent_nodes.add(graph.nodes[adj_node])
            
            # Check and add the nodes from the adjacent squares (top, bottom, left, right, diagonals).
            # Top
            if row - 1 >= min_row:
                if (row - 1, col) in graph.grid:
                    for adj_node in graph.grid[(row - 1, col)]:
                        adjacent_nodes.add(graph.nodes[adj_node])
            # Bottom
            if row + 1 <= max_row:
                if (row + 1, col) in graph.grid:
                    for adj_node in graph.grid[(row + 1, col)]:
                        adjacent_nodes.add(graph.nodes[adj_node])
            # Left
            if col - 1 >= min_col:
                if (row, col - 1) in graph.grid:
                    for adj_node in graph.grid[(row, col - 1)]:
                        adjacent_nodes.add(graph.nodes[adj_node])
            # Right
            if col + 1 <= max_col:
                if (row, col + 1) in graph.grid:
                    for adj_node in graph.grid[(row, col + 1)]:
                        adjacent_nodes.add(graph.nodes[adj_node])
            # Top-left diagonal
            if row - 1 >= min_row and col - 1 >= min_col:
                if (row - 1, col - 1) in graph.grid:
                    for adj_node in graph.grid[(row - 1, col - 1)]:
                        adjacent_nodes.add(graph.nodes[adj_node])
            # Top-right diagonal
            if row - 1 >= min_row and col + 1 <= max_col:
                if (row - 1, col + 1) in graph.grid:
                    for adj_node in graph.grid[(row - 1, col + 1)]:
                        adjacent_nodes.add(graph.nodes[adj_node])
            # Bottom-left diagonal
            if row + 1 <= max_row and col - 1 >= min_col:
                if (row + 1, col - 1) in graph.grid:
                    for adj_node in graph.grid[(row + 1, col - 1)]:
                        adjacent_nodes.add(graph.nodes[adj_node])
            # Bottom-right diagonal
            if row + 1 <= max_row and col + 1 <= max_col:
                if (row + 1, col + 1) in graph.grid:
                    for adj_node in graph.grid[(row + 1, col + 1)]:
                        adjacent_nodes.add(graph.nodes[adj_node])
        
        debug_print(f"Finding FOC and DTWP: Looking at index {i} out of range {num_nodes}, comparing {len(adjacent_nodes)}")
        # Now, iterate through adjacent nodes only
        for node2 in adjacent_nodes:
            if node1 == node2:
                continue  # Skip self comparison
            
            colocation_count, overlap_time, overlap_periods = findFrequencyAndDwellTime(node1.continuous_periods, node2.continuous_periods, min_time_together, radius)
            
            edge = node1.getEdge(node2) or graph.add_edge(node1, node2)

            # Apply minimum time threshold for dwell time
            if overlap_time < min_time_together:  # Convert min_time_together to seconds
                overlap_time = 0

            if overlap_time > 0:
                edge.addOverlapTime(overlap_time)
                edge.addOverlapPeriods(overlap_periods)
            
            if colocation_count > 0:
                edge.addColocationCount(colocation_count)
            
            edge.fixWeight()

            dwell_time_matrix[i][graph.nodes.index(node2)] = overlap_time
            dwell_time_matrix[graph.nodes.index(node2)][i] = overlap_time  # Ensure symmetry

            colocation_matrix[i][graph.nodes.index(node2)] = colocation_count
            colocation_matrix[graph.nodes.index(node2)][i] = colocation_count  # Ensure symmetry

    return colocation_matrix, dwell_time_matrix

def get_grid_square(query, min_point, max_point, width_meters):
    """
    Determines the grid square for a given query point within a bounding box.
    Now returns a tuple (row, col) instead of a single index.

    Parameters:
    - query: Tuple (lat, lon) for the query point.
    - min_point: Tuple (min_lat, min_lon) representing the bottom-left corner of the bounding box.
    - max_point: Tuple (max_lat, max_lon) representing the top-right corner of the bounding box.
    - width_meters: The physical width (and height) of each grid square in meters.

    Returns:
    - A tuple (row, col) indicating which cell of the grid the query falls into.
    """
    query_lat, query_lon = query
    min_lat, min_lon = min_point
    max_lat, max_lon = max_point

    # Approximate conversion factors (meters per degree)
    meters_per_deg_lat = 111320.0
    avg_lat = (min_lat + max_lat) / 2.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(avg_lat))
    
    # Convert the square width from meters to degrees
    delta_lat = width_meters / meters_per_deg_lat
    delta_lon = width_meters / meters_per_deg_lon
    
    # Determine the number of rows and columns in the grid.
    num_rows = math.ceil((max_lat - min_lat) / delta_lat)
    num_cols = math.ceil((max_lon - min_lon) / delta_lon)
    
    # Compute the row and column for the query point.
    # Rows are counted from the top (max_lat) downward.
    row = int((max_lat - query_lat) / delta_lat)
    col = int((query_lon - min_lon) / delta_lon)
    
    # Adjust if the query point lies exactly on the boundary.
    if row >= num_rows:
        row = num_rows - 1
    if col >= num_cols:
        col = num_cols - 1
    
    return (row, col)

def debug_print(message: str) -> None:
    # Move the cursor up one line and clear that line
    sys.stdout.write("\033[F")  # Move up one line
    sys.stdout.write("\033[K")  # Clear the line
    # Print the new message
    print(message)

def preprocess(data, radius):
    """
    Preprocesses the data to determine the minimum and maximum latitude and longitude.
    Also provides real-time updates using debug_print.
    
    Args:
        data (list of lists): Each row contains [ADID, datetime, lat, lon, ..., additional_columns].
    
    Returns:
        tuple: (min_lat, max_lat, min_lon, max_lon)
    """
    debug_print("Preprocessing data...")
    min_lat, max_lat = float('inf'), float('-inf')
    min_lon, max_lon = float('inf'), float('-inf')
    
    for row in data:
        if len(row) < 4:
            continue
        try:
            lat, lon = float(row[3]), float(row[4])
            min_lat, max_lat = min(min_lat, lat), max(max_lat, lat)
            min_lon, max_lon = min(min_lon, lon), max(max_lon, lon)
        except ValueError:
            continue
    
    grid = build_grid((min_lat, min_lon), (max_lat, max_lon), radius)

    debug_print("Preprocessing complete.\n")

    return grid, min_lat, max_lat, min_lon, max_lon

def build_grid(min_point, max_point, width_meters):
    """
    Builds the grid based on the bounding box and returns a dictionary mapping each cell 
    (row, col) to a list of its adjacent cells (including diagonals).

    Parameters:
    - min_point: Tuple (min_lat, min_lon) for the bottom-left of the bounding box.
    - max_point: Tuple (max_lat, max_lon) for the top-right of the bounding box.
    - width_meters: The physical width (and height) of each grid square in meters.

    Returns:
    - grid: Dictionary with keys as (row, col) and values as lists of adjacent (row, col) tuples.
    - num_rows: Total number of rows in the grid.
    - num_cols: Total number of columns in the grid.
    """
    min_lat, min_lon = min_point
    max_lat, max_lon = max_point

    # Use the same conversion factors as in get_grid_square.
    meters_per_deg_lat = 111320.0
    avg_lat = (min_lat + max_lat) / 2.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(avg_lat))
    
    delta_lat = width_meters / meters_per_deg_lat
    delta_lon = width_meters / meters_per_deg_lon

    num_rows = math.ceil((max_lat - min_lat) / delta_lat)
    num_cols = math.ceil((max_lon - min_lon) / delta_lon)

    grid = {}
    for row in range(num_rows):
        for col in range(num_cols):
            # For each cell, compute its adjacent cells (neighbors) including diagonals.
            neighbors = []
            for r in range(row - 1, row + 2):
                for c in range(col - 1, col + 2):
                    # Skip the cell itself.
                    if (r, c) == (row, col):
                        continue
                    # Check grid boundaries.
                    if 0 <= r < num_rows and 0 <= c < num_cols:
                        neighbors.append((r, c))
            grid[(row, col)] = []
    return grid
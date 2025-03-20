import sys
import numpy as np
from tests.AY25.testgraph import process_data
from agile.graphing import createGraph, connectNodes, get_grid_square
from agile.classifier import classifyEdges
from visual_graph import generate_visualization
import time 

# Default parameters
DEFAULT_CSV_FILE = "data/test_location_data_gh.csv"
DEFAULT_MIN_TIME_TOGETHER = 3
DEFAULT_MAX_TIME_DIFF = 10
DEFAULT_RADIUS = 50
DEFAULT_ADID = None

def start_visualization(csv_file, min_time_together, max_time_diff, radius, adid):
    """
    Processes the CSV file, creates a graph, connects nodes, and generates an interactive visualization.

    Args:
        csv_file (str): Path to the CSV file.
        min_time_together (int): Minimum time spent together for connection.
        max_time_diff (int): Maximum time difference allowed between interactions.
        radius (int): The spatial radius for considering a connection.

    Returns:
        tuple: The graph object and adjacency matrix.
    """
    data, df = process_data(csv_file)
    print(df)  # Print dataframe for debugging
    graph = createGraph(data, radius)
    connectNodes(graph, 1, min_time_together, max_time_diff, radius, adid, False)
    adj_matrix = np.nan_to_num(graph.adjacency_matrix, nan=0.0)

    generate_visualization(graph, adj_matrix)

    return graph, adj_matrix

def start_visualization_timed(csv_file, min_time_together, max_time_diff, radius, adid):
    """
    Processes the CSV file, creates a graph, and connects nodes. The function does not generate a visualization.
    
    Args:
        csv_file (str): Path to the CSV file.
        min_time_together (int): Minimum time spent together for connection.
        max_time_diff (int): Maximum time difference allowed between interactions.
        radius (int): The spatial radius for considering a connection.

    Returns:
        tuple: The graph object and adjacency matrix.
    """
    start_time = time.time()  # Start the timer

    data, df = process_data(csv_file)
    print(df)  # Print dataframe for debugging
    graph = createGraph(data, radius)
    connectNodes(graph, 1, min_time_together, max_time_diff, radius, adid, False)
    adj_matrix = np.nan_to_num(graph.adjacency_matrix, nan=0.0)

    # Remove the visualization generation part
    # generate_visualization(graph, adj_matrix)  # This line is now commented out

    end_time = time.time()  # End the timer
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")  # Print the time it took

    return graph, adj_matrix


def start_classifier(csv_file=DEFAULT_CSV_FILE, min_time_together=DEFAULT_MIN_TIME_TOGETHER, max_time_diff=DEFAULT_MAX_TIME_DIFF, radius=DEFAULT_RADIUS):
    """
    Processes the CSV file, creates a graph, connects nodes, and classifies edges.

    Args:
        csv_file (str): Path to the CSV file.
        min_time_together (int): Minimum time spent together for connection.
        max_time_diff (int): Maximum time difference allowed between interactions.
        radius (int): The spatial radius for considering a connection.
    """
    data, df = process_data(csv_file)
    print(df)
    graph = createGraph(data, radius)
    connectNodes(graph, 1, df, min_time_together, max_time_diff, radius, False)
    classifyEdges(graph, 50)

# Check if command-line arguments are provided
if len(sys.argv) > 1:
    # Read command-line arguments
    csv_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV_FILE
    min_time_together = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_MIN_TIME_TOGETHER
    max_time_diff = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_MAX_TIME_DIFF
    radius = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_RADIUS
    adid = sys.argv[5] if len(sys.argv) > 5 else DEFAULT_ADID

    print(f"Running with parameters: {csv_file}, {min_time_together}, {max_time_diff}, {radius}")

else:
    # Use default values
    csv_file = DEFAULT_CSV_FILE
    min_time_together = DEFAULT_MIN_TIME_TOGETHER
    max_time_diff = DEFAULT_MAX_TIME_DIFF
    radius = DEFAULT_RADIUS
    adid = DEFAULT_ADID
    print(f"Running with default parameters: {csv_file}, {min_time_together}, {max_time_diff}, {radius}, {adid}")

# Start visualization
start_visualization_timed(csv_file, min_time_together, max_time_diff, radius, adid)
query = (51.05, 0.0)
min_point = (51.0, -0.5)
max_point = (51.1, 0.5)
width_meters = 1000
#print(get_grid_square(query, min_point, max_point, width_meters))
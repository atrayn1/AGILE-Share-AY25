'''
Things to note:
If the graph gets large, rendering might be slower. 
Consider using scattergl (Plotly’s WebGL-based scatter) for faster 
rendering if you have a high number of nodes/edges.
[I IMPLEMENTED THIS]
'''

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from networkx.algorithms.community import girvan_newman
from tests.AY25.testgraph import process_data
from agile.graphing import createGraph, connectNodes, Graph

def most_central_node(adj_matrix):
    """
    Finds the most central node in the adjacency matrix using betweenness centrality.
    
    Args:
        adj_matrix (numpy.ndarray): The adjacency matrix of the graph.
    
    Returns:
        int: The index of the most central node.
    """
    G = nx.from_numpy_array(adj_matrix)
    centrality = nx.betweenness_centrality(G)
    return max(centrality, key=centrality.get)

def generate_visualization(graph, adj_matrix, output_file="interactive_network_graph.html"):
    """
    Generates a visualization of the graph using Plotly.

    Args:
        graph (Graph): The graph object containing nodes and edges.
        adj_matrix (numpy.ndarray): The adjacency matrix of the graph.
        output_file (str): The file path to save the visualization as an HTML file.
    """
    G = nx.from_numpy_array(adj_matrix)  # Convert adjacency matrix to NetworkX graph
    people = graph.get_node_names()  # Get node names (ADIDs)
    
    # Center the graph around the most central node
    center_node = most_central_node(adj_matrix)
    pos = nx.spring_layout(G, seed=42)  # Initial layout
    pos[center_node] = (0, 0)  # Set center node at (0,0)

    # Adjust node positions relative to the center
    x_offset, y_offset = pos[center_node]
    for node in pos:
        pos[node] = (pos[node][0] - x_offset, pos[node][1] - y_offset)

    # Extract node positions
    nodes_x = [pos[node][0] for node in G.nodes()]
    nodes_y = [pos[node][1] for node in G.nodes()]

    # Node sizes based on centrality
    centrality = nx.betweenness_centrality(G)
    node_sizes = [centrality[node] * 100 for node in G.nodes()]

    
    # Prepare hover info with all Node details
    hover_info = [
        (
            f"ADID: {node.adid}<br>"
            f"Neighbors: {', '.join(n.adid for n in node.neighbors) if node.neighbors else 'None'}<br>"
            f"Number of Edges: {len(node.edges)}<br>"
            f"Number of Continuous Periods: {len(node.continuous_periods)}<br>"
            f"Continuous Periods:<br>" + 
            "<br>".join(str(period).replace("Timestamp(", "").replace(")", "").replace("np.float64(", "").strip() 
                        for period in node.continuous_periods if period is not None)
        )
        for node in graph.nodes
    ]

    nodes = go.Scattergl(
        x=nodes_x, y=nodes_y,
        mode='markers+text',
        name='People',
        text=[node.adid for node in graph.nodes],  # Display ADID as node label
        textposition='top center',
        hovertext=hover_info,  # Unique hover text per node
        marker=dict(
            size=10,  # Fixed size for all nodes
            color='blue',
            colorscale='Viridis',
            line=dict(color='black', width=1)
        )
    )

    # Extract edges for visualization
    edges_x = []
    edges_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edges_x.extend([x0, x1, None])  # None for line breaks
        edges_y.extend([y0, y1, None])

    # Create the edges line plot
    edges = go.Scattergl(
        x=edges_x, y=edges_y,
        mode='lines',
        name='Connections',
        line=dict(width=1, color='gray')
    )

    # Generate the final plot
    fig = go.Figure(data=[edges, nodes])
    fig.update_layout(
        title="Human Network Graph",
        showlegend=True,
        legend_title="Communities",
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    fig.show()

    # Save the interactive plot as an HTML file
    fig.write_html(output_file)
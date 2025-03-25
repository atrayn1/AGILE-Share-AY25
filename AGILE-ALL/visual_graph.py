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

def generate_visualization(graph, adj_matrix, top_adids, output_file="interactive_network_graph.html", min_weight=None, max_weight=None):
    """
    Generates a visualization of the graph using Plotly, filtered by top_adids and optionally by weight range.

    Args:
        graph (Graph): The graph object containing nodes and edges.
        adj_matrix (numpy.ndarray): The adjacency matrix of the graph.
        top_adids (list): List of specific ADIDs to display.
        output_file (str): The file path to save the visualization as an HTML file.
        min_weight (float, optional): Minimum weight of edges to include. Defaults to None.
        max_weight (float, optional): Maximum weight of edges to include. Defaults to None.
    """
    print("top, ", top_adids)
    # Create a mapping from ADID to index
    adid_to_index = {node.adid: i for i, node in enumerate(graph.nodes)}

    # Filter nodes based on top_adids
    if top_adids:
        selected_indices = {adid_to_index[adid] for adid in top_adids if adid in adid_to_index}
    else:
        selected_indices = set(range(len(graph.nodes)))  # Include all nodes if no filter is applied

    # Create a new NetworkX graph with only selected nodes and edges
    G = nx.Graph()
    
    for i in selected_indices:
        G.add_node(i)  # Add only selected nodes

    # Add edges with the weight condition
    for i in selected_indices:
        for j in selected_indices:
            if i != j and adj_matrix[i, j] > 0:
                weight = adj_matrix[i, j]
                if (min_weight is None or weight >= min_weight) and (max_weight is None or weight <= max_weight):
                    G.add_edge(i, j, weight=weight)

    # Identify isolated nodes (nodes with no edges)
    isolated_nodes = [node for node in G.nodes if G.degree(node) == 0]
    G.remove_nodes_from(isolated_nodes)  # Remove them from visualization

    # Convert remaining nodes to ADIDs
    people = [graph.nodes[i].adid for i in G.nodes]

    # Center the graph around the most central node
    if G.nodes:
        center_node = most_central_node(nx.to_numpy_array(G))
        pos = nx.spring_layout(G, seed=42, k=0.1)  # Set 'k' for even spacing
        pos[center_node] = (0, 0)  # Set center node at (0,0)
    else:
        pos = {}

    # Extract node positions
    nodes_x = [pos[node][0] for node in G.nodes]
    nodes_y = [pos[node][1] for node in G.nodes]

    # Prepare hover info
    hover_info = [
        (
            f"ADID: {graph.nodes[node].adid}<br>"
            f"Neighbors: {', '.join(n.adid for n in graph.nodes[node].neighbors if n.adid in top_adids) if graph.nodes[node].neighbors else 'None'}<br>"
            f"Number of Edges: {len([n for n in graph.nodes[node].neighbors if n.adid in top_adids])}<br>"
            f"Number of Continuous Periods: {len(graph.nodes[node].continuous_periods)}<br>"
            f"Continuous Periods:<br>" + 
            "<br>".join(str(period).replace("Timestamp(", "").replace(")", "").replace("np.float64(", "").strip() 
                        for period in graph.nodes[node].continuous_periods if period is not None)
        )
        for node in G.nodes
    ]

    nodes = go.Scatter(
        x=nodes_x, y=nodes_y,
        mode='markers+text',
        name='People',
        text=people,
        textposition='top center',
        hovertext=hover_info,
        marker=dict(
            size=10,
            color='blue',
            colorscale='Viridis',
            line=dict(color='black', width=1)
        )
    )

    # Extract edges for visualization
    edges_x = []
    edges_y = []
    edge_hover_x = []
    edge_hover_y = []
    edge_hover_text = []
    
    for edge in G.edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edges_x.extend([x0, x1, None])  
        edges_y.extend([y0, y1, None])
        
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        edge_hover_x.append(mid_x)
        edge_hover_y.append(mid_y)

        graph_edge = graph.get_edge(graph.nodes[edge[0]], graph.nodes[edge[1]])
        if graph_edge:
            edge_hover_text.append(
                f"Weight: {graph_edge.weight}<br>"
                f"Connecting {graph_edge.node1.adid} to {graph_edge.node2.adid}<br>"
                f"Colocation Count: {graph_edge.colocation_count}<br>"
                f"Overlap Time: {graph_edge.overlap_time}<br>"
                f"Overlap Periods:<br>" + "<br>".join(map(str, graph_edge.overlap_periods))
            )
        else:
            edge_hover_text.append("No data")

    edges = go.Scattergl(
        x=edges_x, y=edges_y,
        mode='lines',
        name='Connections',
        line=dict(width=1, color='gray'),
        hoverinfo='none'
    )

    edge_hover = go.Scatter(
        x=edge_hover_x, y=edge_hover_y,
        mode='markers',
        marker=dict(size=5, color='rgba(0,0,0,0)'),
        hoverinfo='text',
        hovertext=edge_hover_text
    )

    fig = go.Figure(data=[edges, edge_hover, nodes])
    fig.update_layout(
        title="Filtered ADID Dataset Graph",
        showlegend=True,
        legend_title="Communities",
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    fig.write_html(output_file)
    return fig
    
import torch
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from agile.graphing import createGraph
from tests.AY25.testgraph import process_data

def create_networkx_graph(graph, node_labels=None):
    """
    Converts the custom Graph object into a NetworkX graph.

    Args:
        graph (Graph): The custom Graph object.
        node_labels (np.ndarray, optional): Node labels, if available.

    Returns:
        nx.Graph: A NetworkX graph representing the custom Graph object.
    """
    # Create an empty NetworkX graph
    G = nx.Graph()

    # Add nodes with features from the Graph object
    for idx, node in enumerate(graph.get_nodes()):
        node_id = idx  # Use index as node ID
        node_features = node.features if node.features is not None else []  # Assuming features are stored in 'features'
        
        G.add_node(node_id, feature=node_features)  # Add node with features

        # Optionally add node labels
        if node_labels is not None:
            G.nodes[node_id]['label'] = node_labels[idx]

    # Add edges from the adjacency matrix
    for i in range(graph.num_nodes):
        for j in range(i + 1, graph.num_nodes):  # Only add upper triangle to avoid duplication
            weight = graph.adjacency_matrix[i, j].item()  # Get the edge weight
            if weight != 0:  # Only add edges that have a non-zero weight
                G.add_edge(i, j, weight=weight)

    return G


def plot_graph_with_plotly(graph_data, node_labels=None):
    """
    Visualizes a NetworkX graph using Plotly.
    
    Args:
        graph_data (torch_geometric.data.Data): The PyTorch Geometric Data object.
        node_labels (np.ndarray, optional): Node labels for visualization, if available.
    """
    # Convert PyTorch Geometric Data to NetworkX graph
    nx_graph = create_networkx_graph(graph_data, node_labels)

    # Get positions for each node using a layout algorithm (e.g., spring_layout)
    pos = nx.spring_layout(nx_graph, seed=42)  # You can change the layout algorithm if needed

    # Extract node positions for Plotly visualization
    node_x = [pos[node][0] for node in nx_graph.nodes()]
    node_y = [pos[node][1] for node in nx_graph.nodes()]

    # Create a scatter plot for the nodes
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=20,
            color=[node_labels[i] if node_labels is not None else 0 for i in range(len(node_x))],
            colorbar=dict(thickness=15, title='Node Label', xanchor='left', titleside='right')
        )
    )

    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in nx_graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=0.5, color='#888'),
        hoverinfo='none'
    )

    # Create the Plotly figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="NetworkX Graph Visualization",
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)
                    ))

    fig.show()

# Path to the CSV file
csv_file = "data/4adids_dwelltime.csv"

# Read the CSV file using pandas
data, df = process_data(csv_file)
print(df)
# Create the graph
graph = createGraph(data)

nx_graph = create_networkx_graph(graph)

# Visualize the graph using Plotly
plot_graph_with_plotly(nx_graph)

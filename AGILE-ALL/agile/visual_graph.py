# This will make the visualization of the graph

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from networkx.algorithms.community import girvan_newman

# Define an adjacency matrix (Example: 4 people)
# need this matrix to be the adjacency matrix we actually use
adj_matrix = np.array([[0, 1862, 37978, 0],
[1862, 0, 1606, 37978],
[37978, 1606, 0, 0],
[0, 37978, 0, 0]])

# Create a NetworkX graph from the adjacency matrix
G = nx.from_numpy_array(adj_matrix)

# Assume these are the people represented by each row of the adjacency matrix
people = ["adid_1", "adid_2", "adid_3", "adid_4"]

# Create a layout for the graph (position of nodes)
#pos = nx.spring_layout(G, k=0.5, seed=42)  # k controls the "spread" of nodes
#pos = nx.circular_layout(G)  # Creates a circular layout
pos = nx.spectral_layout(G)  # Uses the spectral layout, often good for community detection

# Adjust the layout based on edge weights (stronger edges pull nodes closer)
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    
    # Adjust the distance based on edge weight (higher weight -> closer nodes)
    edge_weight = adj_matrix[edge[0], edge[1]]
    
    # You can use a scaling factor to control how much the edge weight affects the distance
    scaling_factor = 0.001  # Experiment with this value to control the effect
    distance = edge_weight * scaling_factor
    
    # Update positions to pull nodes closer for stronger edges
    pos[edge[0]] = (x0 + distance, y0 + distance)
    pos[edge[1]] = (x1 - distance, y1 - distance)


# Extract node positions
nodes_x = [pos[node][0] for node in G.nodes()]
nodes_y = [pos[node][1] for node in G.nodes()]

# Calculate node size based on degree (connections)
node_sizes = [G.degree(node) * 10 for node in G.nodes()]  # Degree-based size scaling

# not sure how to use this:
centrality = nx.betweenness_centrality(G)  # or closeness_centrality, degree_centrality, etc.
#node_sizes = [centrality[node] * 5000 for node in G.nodes()]

# Detect communities (optional: community detection)
communities = girvan_newman(G)
first_community = next(communities)  # Get the first split (you can adjust this)

# Assign colors to communities
community_colors = {node: i for i, community in enumerate(first_community) for node in community}
node_colors = [community_colors[node] for node in G.nodes()]

# Assign node shapes based on community
community_shapes = {0: 'circle', 1: 'square', 2: 'diamond', 3: 'triangle', 4: 'star', 5: 'hexagon', 6: 'pentagon', 7: 'x', 8: 'bowtie'}  # Define shape for each community
node_shapes = [community_shapes[community_colors[node]] for node in G.nodes()]

# Prepare hover info (additional details when hovering over nodes)
hover_info = [f"Name: {name}<br>Connections: {G.degree(name)}" for name in people]

# Create the nodes scatter plot (representing people)
nodes = go.Scatter(
    x=nodes_x, y=nodes_y,
    mode='markers+text',
    name='People',
    text=people,  # Labels for each node (person)
    textposition='top center',
    hovertext=hover_info,  # Add more info on hover
    marker=dict(
        size=node_sizes,  # Vary size based on node degree
        color=node_colors,  # Color based on community
        colorscale='Viridis',  # Color scale for communities
        colorbar=dict(title="Community"),
        symbol=node_shapes,
        line=dict(color='black', width=1)
    )
)

# Extract edges data for plotting
edges_x = []
edges_y = []

for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edges_x.append(x0)
    edges_y.append(y0)
    edges_x.append(x1)
    edges_y.append(y1)

# Create the edges line plot (representing connections between people)
edges = go.Scatter(
    x=edges_x, y=edges_y,
    mode='lines',
    name='Connections',
    line=dict(width=1, color='gray')
)

# Combine nodes and edges into one Plotly figure
fig = go.Figure(data=[edges, nodes])

# Customize the layout of the graph
fig.update_layout(
    title="Human Network Graph",
    showlegend=True,
    legend_title="Communities",
    legend=dict(x=0.85, y=0.95),
    hovermode='closest',
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=False, zeroline=False)
)

# Show the interactive plot
fig.show()

# To add the plot to the existing AGILE webpage
# I want this html file to go straight to the 
# visual_graphs directory
#fig.write_html("interactive_network_graph.html")
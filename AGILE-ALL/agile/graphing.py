import torch
import torch.nn as nn
import torch.nn.functional as F

class Graph:
    def __init__(self, num_nodes):
        """
        Initializes the graph.

        Args:
            num_nodes (int): Initial number of nodes in the graph.
        """
        self.num_nodes = num_nodes
        self.adjacency_matrix = torch.zeros((num_nodes, num_nodes))
        self.node_features = torch.zeros((num_nodes, 1))
        self.edge_features = {}

    def add_node(self, features=None):
        """
        Adds a new node to the graph.

        Args:
            features (torch.Tensor, optional): Features of the new node. Defaults to None.
        """
        self.num_nodes += 1
        # Expand adjacency matrix
        new_adj = torch.zeros((self.num_nodes, self.num_nodes))
        new_adj[:-1, :-1] = self.adjacency_matrix
        self.adjacency_matrix = new_adj

        # Expand node features
        if features is None:
            features = torch.zeros((1, self.node_features.shape[1]))
        else:
            assert features.shape[1] == self.node_features.shape[1], "Feature dimension mismatch."
        self.node_features = torch.cat((self.node_features, features), dim=0)

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
        self.node_features = torch.cat((
            self.node_features[:node],
            self.node_features[node + 1:]
        ), dim=0)

        # Remove associated edge features
        self.edge_features = {
            (n1, n2): features
            for (n1, n2), features in self.edge_features.items()
            if n1 != node and n2 != node
        }

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
            features (torch.Tensor): A tensor of shape (num_nodes, feature_dim).
        """
        assert features.shape[0] == self.num_nodes, "Feature size mismatch with number of nodes."
        self.node_features = features

    def set_edge_features(self, node1, node2, features):
        """
        Sets features for a specific edge.

        Args:
            node1 (int): Index of the first node.
            node2 (int): Index of the second node.
            features (torch.Tensor): Features for the edge.
        """
        self.edge_features[(node1, node2)] = features
        self.edge_features[(node2, node1)] = features  # Assuming undirected graph

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

    def forward_pass(self, node_transform=None, edge_transform=None):
        """
        Performs a forward pass on the graph.

        Args:
            node_transform (callable): A function to transform node features.
            edge_transform (callable): A function to transform edge features.

        Returns:
            torch.Tensor: Updated node features.
        """
        updated_features = self.node_features.clone()

        for i in range(self.num_nodes):
            neighbors = self.get_neighbors(i)
            neighbor_features = torch.stack([self.node_features[n] for n in neighbors])

            if edge_transform:
                for n in neighbors:
                    edge_feat = self.edge_features.get((i, n), None)
                    if edge_feat is not None:
                        neighbor_features += edge_transform(edge_feat)

            if node_transform:
                updated_features[i] = node_transform(self.node_features[i], neighbor_features)

        self.node_features = updated_features
        return self.node_features

def createGraph(data):
    """
    Creates a graph from the given data.

    Args:
        data (list of lists): Each row contains [ADID, lat, lon, ...].
            ADIDs are used to define nodes.

    Returns:
        Graph: A constructed graph with nodes and features.
    """
    # Extract unique ADIDs and map them to indices
    unique_adids = list(set(row[0] for row in data))
    adid_to_index = {adid: idx for idx, adid in enumerate(unique_adids)}

    num_nodes = len(unique_adids)
    graph = Graph(num_nodes)

    # Add node features (latitude and longitude)
    features = torch.zeros((num_nodes, 2))  # Assuming features are lat and lon
    for row in data:
        adid, lat, lon = row[:3]
        node_idx = adid_to_index[adid]
        features[node_idx] = torch.tensor([lat, lon])

    graph.set_node_features(features)

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
        int: The time spent at the same place. Currently does nothing.
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

if __name__ == "__main__":
    createGraph()

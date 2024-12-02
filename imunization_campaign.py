import networkx as nx
import random
import matplotlib.pyplot as plt

class ImmunizationCampaign:
    def __init__(self, graph, seed=None):
        self.graph = graph
        self.immunized_nodes = []  
        self.neighbors_of_immunized = []
        if seed is not None:
            random.seed(seed)

    def immunize(self, method="random", num_nodes=10, infected_nodes=None):

        if method == "random":
            nodes_to_immunize = random.sample(self.graph.nodes, num_nodes)
        elif method == "hubs":
            nodes_to_immunize = sorted(self.graph.degree, key=lambda x: x[1], reverse=True)[:num_nodes]
            nodes_to_immunize = [node for node, _ in nodes_to_immunize]
        elif method == "neighbors" and infected_nodes:
            neighbors = set()
            for node in infected_nodes:
                neighbors.update(self.graph.neighbors(node))
            nodes_to_immunize = list(neighbors)[:num_nodes]
        else:
            raise ValueError("Invalid immunization method or 'infected_nodes' not provided for 'neighbors'.")

        self.immunized_nodes = nodes_to_immunize

        self.neighbors_of_immunized = set()
        for node in nodes_to_immunize:
            self.neighbors_of_immunized.update(self.graph.neighbors(node))
        self.neighbors_of_immunized -= set(self.immunized_nodes)  

        self.graph.remove_nodes_from(nodes_to_immunize)
        return nodes_to_immunize

    def plot_graph(self, title="Graph After Immunization"):
    
        pos = nx.spring_layout(self.graph, seed=42)  
        node_colors = []
        for node in self.graph.nodes:
            if node in self.immunized_nodes:
                node_colors.append("green")
            elif node in self.neighbors_of_immunized:
                node_colors.append("blue")
            else:
                node_colors.append("orange")

        plt.figure(figsize=(10, 8))
        nx.draw(
            self.graph,
            pos,
            node_color=node_colors,
            with_labels=True,
            node_size=300,
            edge_color="gray",
            alpha=0.7,
        )
        plt.title(title)
        plt.show()

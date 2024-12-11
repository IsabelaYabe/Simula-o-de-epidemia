"""
Module: SISModel for Simulating the Spread of Infection in a Network

This module provides the `SISModel` class for simulating the Susceptible-Infected-Susceptible (SIS) epidemic model
on a graph. It includes options for introducing immunization campaigns, running the model over a series of time steps,
and visualizing the results with animations or static plots.

Author: Isabela Yabe
Last Modified: 02/12/2024
Status: Complete

Dependencies:
    - networkx
    - numpy
    - random
    - matplotlib.pyplot
    - matplotlib.animation
"""

import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from immunization_campaign import ImmunizationCampaign

class SISModel:
    """
    A class to simulate the Susceptible-Infected-Susceptible (SIS) model of infection spread on a network.

    The SIS model assumes nodes can transition between two states:
    - Susceptible (S): A node that is not currently infected but can become infected.
    - Infected (I): A node that is currently infected and can infect neighbors or recover.

    Attributes:
        graph (networkx.Graph): The network representing the nodes and their connections.
        beta (float): Infection rate, the probability of a susceptible node being infected by an infected neighbor.
        mu (float): Recovery rate, the probability of an infected node recovering in a time step.
        time_step (int): The current time step in the simulation.
        history (list[int]): A list tracking the number of infected nodes at each time step.
        states (dict): A dictionary mapping each node to its current state (0 for susceptible, 1 for infected).
        campaign (ImmunizationCampaign): Optional immunization campaign applied before infection spread.
        start_immunize (int): Number of nodes to immunize at the beginning of the simulation.
    """
    def __init__(self, graph, beta, mu, initial_infected=5, immunization_campaign=None, start_immunize=5, seed=None):
        """
        Initializes the SISModel simulation.

        Args:
            graph (networkx.Graph): The network on which the simulation runs.
            beta (float): Infection rate.
            mu (float): Recovery rate.
            initial_infected (int): Number of nodes initially infected. Default is 5.
            immunization_campaign (ImmunizationCampaign, optional): Instance of an immunization campaign. Default is None.
            start_immunize (int): Number of nodes to immunize at the beginning. Default is 5.
            seed (int, optional): Seed for random number generation. Default is None.

        Raises:
            ValueError: If there are not enough unimmunized nodes to initiate infection.
        """        
        self.graph = graph
        self.beta = beta
        self.mu = mu
        self.time_step = 0
        self.history = []
        self.states = {node: 0 for node in graph.nodes}
        self.campaign = immunization_campaign  
        self.start_immunize = start_immunize

        if seed is not None:
            random.seed(seed)

        eligible_nodes = list(set(self.graph.nodes()))
        if self.campaign:
            self.campaign.immunize(self.start_immunize)
            eligible_nodes = list(set(eligible_nodes) - set(self.campaign.immunized_nodes))

        
            if len(eligible_nodes) < initial_infected:
                raise ValueError("There are not enough unimmunized nodes to initiate infection")

        initial_infected_nodes = random.sample(eligible_nodes, initial_infected)
        for node in initial_infected_nodes:
            self.states[node] = 1

        self.history.append(initial_infected)

    def nodes_actives(self):
        """
        Retrieves the list of nodes that are active in the simulation (not immunized).

        Returns:
            list: Active nodes in the simulation.
        """
        if self.campaign:
            return list(set(self.graph.nodes()) - set(self.campaign.immunized_nodes))
        else:
            return list(self.graph.nodes())

    def step(self):
        """
        Executes one time step of the simulation.

        Updates the states of nodes based on infection and recovery rules and tracks the number of infected nodes.
        """
        new_states = self.states.copy()
        for node in self.nodes_actives():
            if self.states[node] == 1:  
                if random.random() < self.mu:
                    new_states[node] = 0
    
                for neighbor in self.graph.neighbors(node):
                    if (not self.campaign or neighbor not in self.campaign.immunized_nodes) and self.states[neighbor] == 0:
                        if random.random() < self.beta:
                            new_states[neighbor] = 1
    
        self.states = new_states
        new_history_data = sum(self.states.values())  
        self.history.append(new_history_data)
        self.time_step += 1

    def run(self, steps=50):
        """
        Runs the simulation for a specified number of steps.

        Args:
            steps (int): Number of time steps to simulate. Default is 50.
        """
        for _ in range(steps):
            self.step()

    def plot(self, title="Fraction of infected nodes over time"):
        """
        Plots the fraction of infected nodes over time.

        The x-axis represents time steps, and the y-axis represents the fraction of infected nodes.
        """
        total_nodes = len(self.graph.nodes())
        fraction_infected = [infected / total_nodes for infected in self.history]

        plt.figure(figsize=(8, 5))
        plt.plot(fraction_infected, label="Fraction Infected", color="red")
        plt.xlabel("Time Steps")
        plt.ylabel("Fraction of Infected Nodes")
        plt.title("Evolução do Modelo SIS (Fração de Infectados)")
        plt.legend()
        plt.title(title)
        plt.show()
    
    def animate(self, steps=100, interval=500):
        """
        Animates the SIS model on the graph.

        Args:
            steps (int): Number of time steps to run the animation. Default is 100.
            interval (int): Interval in milliseconds between frames. Default is 500.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(self.graph, seed=42)  
        
        def animation_frame(frame):
            ax.clear()
            self.step()  
            legend_elements = []
            node_colors = ["orange"] * len(self.graph.nodes)
            if self.campaign:
                node_colors = []
                neighbors_of_immunized = self.campaign.neighbors_of_immunized
            
                for node in self.graph.nodes():
                    if node in self.campaign.immunized_nodes:  
                        node_colors.append("green")
                        
                    elif node in neighbors_of_immunized:  
                        node_colors.append("blue")
                    else: 
                        node_colors.append("orange")
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label='Immunized Nodes', markerfacecolor='green', markersize=10))
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label='Neighbors of Immunized', markerfacecolor='blue', markersize=10))    
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label='Susceptible Nodes', markerfacecolor='orange', markersize=10))

            infected_nodes = [node for node in self.graph.nodes() if self.states[node] == 1]
            susceptible_nodes = [node for node in self.graph.nodes() if self.states[node] == 0]

            nx.draw_networkx_nodes(
                self.graph,
                pos,
                nodelist=susceptible_nodes,
                node_color=[node_colors[n] for n in susceptible_nodes],
                node_shape="o",
                ax=ax
            )
            nx.draw_networkx_nodes(
                self.graph,
                pos,
                nodelist=infected_nodes,
                node_color=[node_colors[n] for n in infected_nodes],
                node_shape="*",
                ax=ax
            )
            nx.draw_networkx_edges(self.graph, pos, ax=ax, alpha=0.5)

            ax.set_title(f"Time Step: {frame + 1}\nInfected: {self.history[-1]}")
            
            legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', label='Infected Nodes', markerfacecolor='red', markersize=10)) 
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label='Not infected Nodes', markerfacecolor='red', markersize=10)) 

            ax.legend(handles=legend_elements, loc="upper right", title="Node Types")
            ax.axis("off")


        ani = animation.FuncAnimation(fig, animation_frame, frames=steps, interval=interval, repeat=False)
        plt.show()

if __name__ == "__main__":
    #n = 100
    #k_avg = 10
    #p = k_avg / (n - 1)
    #G = nx.erdos_renyi_graph(n, p, seed=42)
    #
    #sis = SISModel(G, beta=0.3, mu=0.1, initial_infected=1, seed=42)
    #sis.animate(steps=100, interval=300)
    #sis.plot()
    #
    #campaign = ImmunizationCampaign(graph=G, method="hubs", seed=42)
    #sis_with_immunization = SISModel(G, beta=0.3, mu=0.1, initial_infected=1, immunization_campaign=campaign, seed=42)
    #sis_with_immunization.animate(steps=100, interval=300)
    #sis_with_immunization.plot()

    N = 100
    k_medio = 10  
    initial_infected = 5  
    beta = 0.01 
    mu_values = [0.1, 0.2, 0.3] 
    m = k_medio // 2 

    max_immunized = 300 
    steps = 100

    G = nx.barabasi_albert_graph(N, m)

    methods = ["random", "hubs", "neighbors"]
    campaign = ImmunizationCampaign(graph=G, method=methods[1], seed=42)
    sis_with_immunization = SISModel(G, beta=0.3, mu=0.1, initial_infected=1, immunization_campaign=campaign, seed=42)
    sis_with_immunization.animate(steps=100, interval=300)
    sis_with_immunization.plot()
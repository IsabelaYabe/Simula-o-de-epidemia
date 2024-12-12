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
from copy import deepcopy
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
    def __init__(self, graph, beta, mu, initial_infected=5, immunization_campaign=None, start_immunize=None, seed=None):
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
        self.graph = deepcopy(graph)
        self.beta = beta
        self.mu = mu
        self.time_step = 0
        self.states = {node: 0 for node in self.graph.nodes()}
        self.history = []
        self.campaign = immunization_campaign  
        self.start_immunize = start_immunize

        if seed is not None:
            random.seed(seed)
    
        if self.campaign:
            self.campaign.immunize(self.start_immunize)
            for node in self.campaign.immunized_nodes:
                if node in self.states:
                    del self.states[node]
    
        initial_infected_nodes = random.sample(list(self.states.keys()), min(initial_infected, len(self.states)))
        for node in initial_infected_nodes:
            self.states[node] = 1

        self.history.append(len(initial_infected_nodes))

    def step(self):
        new_states = deepcopy(self.states)
        for node in [node for node, value in new_states.items() if value == 1]:
            for neighbor in self.graph.neighbors(node):
                if new_states[neighbor] == 0:
                    if random.random() < self.beta:
                        new_states[neighbor] = 1
            if random.random() < self.mu:
                new_states[node] = 0

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
    
if __name__ == "__main__":
    n = 100
    k_avg = 10
    p = k_avg / (n - 1)
    G = nx.erdos_renyi_graph(n, p, seed=42)

    sis = SISModel(G, beta=0.3, mu=0.1, initial_infected=1, seed=42)
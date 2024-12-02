import networkx as nx
import random
import matplotlib.pyplot as plt
from immunization_campaign import ImmunizationCampaign

class SISModel:
    def __init__(self, graph, beta, mu, initial_infected=5, seed=None):
        self.graph = graph
        self.beta = beta
        self.mu = mu
        self.time_step = 0
        self.history = [initial_infected]
        self.states = {node: 0 for node in graph.nodes}
        
        if seed is not None:
            random.seed(seed)
        
        initial_infected_nodes = random.sample(graph.nodes, initial_infected)
        for node in initial_infected_nodes:
            self.states[node] = 1
        

    def step(self):
        new_states = self.states.copy()
        new_history_data = self.history[-1]
        for node in self.graph.nodes:
            if self.states[node] == 1:  
                if random.random() < self.mu:
                    new_states[node] = 0 
                    new_history_data -= 1
                for neighbor in self.neighbors(node):
                    if (not self.states[neighbor]) and (random.random() < self.beta):
                        new_states[neighbor] = 1
                        new_history_data += 1

        self.states = new_states
        self.history.append(new_history_data)
        self.time_step += 1

    def run(self, steps=50):
        for _ in range(steps):
            self.step()

    def plot(self):
        total_nodes = len(self.graph.nodes)
        fraction_infected = [infected / total_nodes for infected in self.history]

        plt.figure(figsize=(8, 5))
        plt.plot(fraction_infected, label="Fraction Infected", color="red")
        plt.xlabel("Time Steps")
        plt.ylabel("Fraction of Infected Nodes")
        plt.title("Evolução do Modelo SIS (Fração de Infectados)")
        plt.legend()
        plt.show()
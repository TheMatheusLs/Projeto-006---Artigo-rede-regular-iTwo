# Cria a topologia de rede NSFNet usando o NetworkX
import networkx as nx
import numpy as np

from Topology.Generic import Generic
from Enviroment.Settings import IS_COST_TYPE

class iTwo(Generic):
    def __init__(self, num_of_slots: int):

        NUM_OF_NODES = 8

        # Adiciona os enlaces (representando as conexões) (source, destination, length)
        network_lenghts = [
            (0, 1, 100),
            (1, 2, 100), 
            (2, 3, 100),
            (2, 5, 100),
            (3, 4, 100),
            (4, 5, 100),
            (5, 6, 100),
            (6, 7, 100),
        ]

        self.num_of_links = 0
        self.num_of_nodes = NUM_OF_NODES
        self.num_of_slots = num_of_slots
        
        self.graph = nx.Graph()

        # Adiciona os nós (representando os equipamentos)
        self.graph.add_nodes_from(range(self.num_of_nodes))

        # Criando um dicionário com um mapa para acessar o espectro dos links a partir de uma tupla (source, destination)
        self.spectrum_map = {}

        for source, destination, length in network_lenghts:

            if IS_COST_TYPE == 'length':
                link_cost = length
            elif IS_COST_TYPE == 'hops':
                link_cost = 1

            # Adiciona os enlaces (representando as conexões) (source, destination, cost)
            self.graph.add_edge(source, destination, cost = link_cost)

            # Adiciona o ID do enlace no mapa de espectro
            self.spectrum_map[(source, destination)] = self.num_of_links

            self.num_of_links += 1

        # Matriz de links (linhas) e slots (colunas)
        self.all_optical_links = np.zeros((self.num_of_links, self.num_of_slots), dtype=np.bool_)
# Cria a topologia de rede NSFNet usando o NetworkX
import networkx as nx
import numpy as np
import heapq

from Enviroment.Settings import *

class Generic():
    def __init__(self, num_of_nodes: int, num_of_slots: int, lengths: list):

        self.num_of_links = 0
        self.num_of_nodes = num_of_nodes
        self.num_of_slots = num_of_slots
        
        self.graph = nx.Graph()

        # Adiciona os nós (representando os equipamentos)
        self.graph.add_nodes_from(range(self.num_of_nodes))

        # Criando um dicionário com um mapa para acessar o espectro dos links a partir de uma tupla (source, destination)
        self.spectrum_map = {}

        for source, destination, length in lengths:

            if IS_COST_TYPE == 'length':
                link_cost = length
            elif IS_COST_TYPE == 'hops':
                link_cost = 1

            # Adiciona os enlaces (representando as conexões) (source, destination, cost)
            self.graph.add_edge(source, destination, cost = link_cost)

            # Adiciona o ID do enlace no mapa de espectro
            self.spectrum_map[(source, destination)] = self.num_of_links

            self.num_of_links += 1

            # Adiciona os enlaces (representando as conexões) (destination, source, cost)
            self.graph.add_edge(destination, source, cost = link_cost)

            self.spectrum_map[(destination, source)] = self.num_of_links
            
            self.num_of_links += 1

        # Matriz de links (linhas) e slots (colunas)
        self.all_optical_links = np.zeros((self.num_of_links, self.num_of_slots), dtype=np.bool_)

    def clean_links(self):
        
        # Limpa todos os links colocando-os como False sem mudar o endereco de memoria
        self.all_optical_links.fill(False)


    def __str__(self):
        return_text = "Links\t Slots\n"

        for i in range(self.num_of_links):
            return_text += f"{i}\t {[status for status in self.all_optical_links[i]]}\n"

        return return_text


    def get_num_of_nodes(self):
        return self.num_of_nodes
    
    
    def get_num_of_links(self):
        return self.num_of_links
    

    def get_num_of_slots(self):
        return self.num_of_slots
    

    def get_all_optical_links(self):
        return self.all_optical_links
    

    def get_graph(self):
        return self.graph
    

    # Algoritmo de roteamento (usando menor caminho)
    def dijkstra_routing(self, source, destination):
        return nx.shortest_path(self.graph, source, destination, weight='cost')


    def k_shortest_paths(self, source, destination, k_routes):
        paths = []
        heap = [(0, [source])]
        
        while heap and len(paths) < k_routes:
            (cost, path) = heapq.heappop(heap)
            current_node = path[-1]
            
            if current_node == destination:
                paths.append((cost, path))
            else:
                for next_node in self.graph[current_node]:
                    if next_node not in path:
                        heapq.heappush(heap, (cost + self.graph[current_node][next_node]['cost'], path + [next_node]))
        
        only_paths = []
        for i in range(len(paths)):
            only_paths.append(paths[i][1] if len(paths[i][1]) > 1 else None)

        return only_paths
    

    def print_graph(self):

        import matplotlib.pyplot as plt

        # Plotar o grafo
        nx.draw(self.graph, pos=self.node_positions, with_labels=True, node_size=500, node_color='skyblue', font_size=8, font_color='black', font_weight='bold')
        plt.title("Topology Generic")
        plt.show()


    def allocate_slots(self, route, slots):

        for link_index in route._path_uplink:
            for slot in slots:
                self.all_optical_links[link_index, slot] = True

        # TODO: Verificar se é necessário alocar os slots do downlink
    
    def deallocate_slots(self, route, slots):

        for link_index in route._path_uplink:
            for slot in slots:
                self.all_optical_links[link_index, slot] = False

        # TODO: Verificar se é necessário desalocar os slots do downlink

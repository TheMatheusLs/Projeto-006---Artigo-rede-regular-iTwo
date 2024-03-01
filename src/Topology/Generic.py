# Cria a topologia de rede NSFNet usando o NetworkX
import networkx as nx
import numpy as np
import heapq

from Enviroment.Settings import *
from Enviroment.OpticalLink import OpticalLink
from Enviroment.OpticalNode import OpticalNode
from Enviroment.route import Route

class Generic():
    def __init__(self, num_of_nodes: int, num_of_slots: int, lengths: list):

        self.num_of_links = 0
        self.num_of_nodes = num_of_nodes
        self.num_of_slots = num_of_slots
        
        self.graph = nx.Graph()

        # Adiciona os nós (representando os equipamentos)
        self.graph.add_nodes_from(range(self.num_of_nodes))

        self.links_adjacency_matrix = {}
        
        self.optical_nodes = [
            OpticalNode(opticalSwitchID=i) for i in range(self.num_of_nodes)
        ]

        # Criando um dicionário com um mapa para acessar o espectro dos links a partir de uma tupla (source, destination)
        self.spectrum_map = {}

        for source, destination, length in lengths:

            if IS_COST_TYPE == 'length':
                link_cost = length
            elif IS_COST_TYPE == 'hops':
                link_cost = 1

            # Adiciona os enlaces (representando as conexões) (source, destination, cost)
            self.graph.add_edge(source, destination, cost = link_cost)

            link = OpticalLink(self.num_of_links, source, destination, 0, length)
            link.set_cost(link_cost)
            self.links_adjacency_matrix[(source, destination)] = link

            # Adiciona o ID do enlace no mapa de espectro
            self.spectrum_map[(source, destination)] = self.num_of_links

            self.num_of_links += 1

            # Adiciona os enlaces (representando as conexões) (destination, source, cost)
            self.graph.add_edge(destination, source, cost = link_cost)

            link = OpticalLink(self.num_of_links, destination, source, 0, length)
            link.set_cost(link_cost)
            self.links_adjacency_matrix[(destination, source)] = link

            self.spectrum_map[(destination, source)] = self.num_of_links
            
            self.num_of_links += 1

        # Matriz de links (linhas) e slots (colunas)
        self.all_optical_links = np.zeros((self.num_of_links, self.num_of_slots), dtype=np.bool_)

        self.route_id = 0


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
    

    def YEN(self, source, destination, k_routes):
        return list(nx.shortest_simple_paths(self.graph, source, destination, weight='cost'))[:k_routes]



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
                
    
    def djistra_on_hand(self, orNode, destination):
        
        k = -1
        numNodes = self.get_num_of_nodes()
        path = []
        invPath = []
        auxLink = None
        routeDJK = None
        networkDisconnected = False

        custoVertice = [float('inf')] * numNodes
        precedente = [-1] * numNodes
        status = [False] * numNodes

        custoVertice[orNode] = 0
        setVertexes = numNodes

        while (setVertexes > 0 and not networkDisconnected):

            min_value = float('inf')

            for i in range(numNodes):
                if (status[i] == False and custoVertice[i] < min_value):
                    min_value = custoVertice[i]
                    k = i

            if k == destination:
                break

            status[k] = True
            setVertexes -= 1
    
            outputLinkFound = False;

            for j in range(numNodes):

                auxLink = self.links_adjacency_matrix.get((k, j))

                if auxLink and auxLink.is_link_working() and self.optical_nodes[auxLink.sourceNode].is_node_working() and self.optical_nodes[auxLink.destinationNode].is_node_working():

                    outputLinkFound = True
                    
                    if( (status[j] == False) and (custoVertice[k] + auxLink.get_cost() < custoVertice[j]) ):
                       custoVertice[j] = (custoVertice[k] + auxLink.get_cost())
                       precedente[j] = k

            if not outputLinkFound:
                networkDisconnected = True
            

        if not networkDisconnected:
            path.append(destination)
            hops = 0
            j = destination
            
            while(j != orNode):
                hops += 1
                if(precedente[j] != -1):
                    path.append(precedente[j])
                    j = precedente[j]
                else:
                    networkDisconnected = True
                    break
            if not networkDisconnected:
                for h in range(hops+1):
                    invPath.append(path[hops-h])

                # Cria a classe da rota com o caminho encontrado
                routeDJK = self.create_route_by_path(invPath)
                

        return routeDJK
    

    def create_route_by_path(self, path):
        # Retorna o índice dos links que compõem o uplink da rota
        route_uplinks = [self.spectrum_map[path[i], path[i + 1]] for i in range(len(path) - 1)]

        if IS_BIDIRECTIONAL:

            node_path_reversed = path[::-1]

            route_downlinks = [self.spectrum_map[node_path_reversed[i], node_path_reversed[i + 1]] for i in range(len(node_path_reversed) - 1)]
        else:
            route_downlinks = []

        uplinks_spectrum = []
        for link_id in route_uplinks:
            uplinks_spectrum.append(self.all_optical_links[link_id])

        route_cost = 0
        for i in range(len(path) - 1):
            route_cost += self.links_adjacency_matrix[(path[i], path[i + 1])].get_cost()

        routeDJK = Route(node_path = path, route_index = self.route_id, uplink_path=route_uplinks, downlink_path=route_downlinks, number_of_slots=self.num_of_slots, uplinks_spectrums=uplinks_spectrum, cost=route_cost)
                        
        self.route_id += 1

        return routeDJK
    

    def YEN_on_hand(self, orNode, destination, k_routes):
        
        # Lista A contém os caminhos já encontrados
        routesYEN = []
        k_value = 1

        # Fila B contém os caminhos candidatos
        candidateRoutes = []

        # Encontra o primeiro caminho mais curto usando Dijkstra
        shortestRoute = self.djistra_on_hand(orNode, destination)

        if shortestRoute is None:
            return None # Se não houver caminho, retorna None
        
        shortestRoute.set_route_index(k_value)
        k_value += 1
        routesYEN.append(shortestRoute) # Adiciona o primeiro caminho na lista de caminhos

        for k in range(1, k_routes): # repete até encontrar K caminhos

            # Obtem o caminho anterior
            previousRoute = routesYEN[k-1]

            routePath = len(previousRoute._path_node)

            for i in range(routePath-1):

                spur_node_ID = previousRoute._path_node[i]
                rootPath = previousRoute._path_node[:i+1]

                removed_nodes = []
                removed_edges = []

                for route in routesYEN:
                    if (i < len(route._path_node)):

                        shortestRoute = route._path_node[:i+1]

                        if rootPath == shortestRoute:
                            next_node = route._path_node[i+1]

                            link = self.links_adjacency_matrix.get((spur_node_ID, next_node))

                            link.set_link_state(False)
                            removed_edges.append(link)

                for node in rootPath:
                    if node != spur_node_ID:

                        optical_switch = self.optical_nodes[node]

                        optical_switch.set_node_state(False)
                        removed_nodes.append(optical_switch)

                spurRoute = self.djistra_on_hand(spur_node_ID, destination)

                if spurRoute is not None:
                    totalRoute = self.create_route_by_path(rootPath + spurRoute._path_node[1:])

                    if totalRoute not in candidateRoutes:
                        candidateRoutes.append(totalRoute)

                # Restaura os nós e enlaces removidos
                for node in removed_nodes:
                    node.set_node_state(True)

                for link in removed_edges:
                    link.set_link_state(True)

            if len(candidateRoutes) == 0:
                break

            new_path = self.get_route_by_cost(candidateRoutes)

            new_path.set_route_index(k_value)
            k_value += 1
            routesYEN.append(new_path)

            candidateRoutes.remove(new_path)

        return routesYEN
    

    def get_route_by_cost(self, candidateRoutes):

        candidate_routes_clone = candidateRoutes.copy()

        routes_order = []

        while (len(routes_order) < 1 and len(candidate_routes_clone) > 0):

            min_cost = float('inf')
            best_route_index = 0

            for r in range(len(candidate_routes_clone)):

                if min_cost > candidate_routes_clone[r].cost:
                    min_cost = candidate_routes_clone[r].cost
                    best_route_index = r   
                
            route = candidate_routes_clone[best_route_index]

            routes_order.append(route)
            candidate_routes_clone.remove(route)

        return routes_order[0]





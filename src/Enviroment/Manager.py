


import numpy as np
import gymnasium as gym
np.random.seed(42)

import Enviroment.utils as utils
from Enviroment.Settings import *

from Topology.iTwo import iTwo
from Enviroment.route import Route
from Enviroment.demand import Demand
import SpectrumAssignment.RSA_FirstFit as RSA_FirstFit
import SpectrumAssignment.SAR_FirstFit as SAR_FistFit
import SpectrumAssignment.MSCL_combinado as MSCL_combinado


class Enviroment(gym.Env):
    """
        Classe para representar o ambiente de simulação. O ambiente é representado por uma topologia de rede, uma lista de demandas ativas, um tempo de simulação, um número de slots, uma lista de rotas possíveis e um mapa de codificação para os estados de origem e destino em one hot encoding.

        O ambiente herda da classe gym.Env, que é a classe base para todos os ambientes do Gym. Ela expõe os seguintes métodos:
            - step(action): Executa uma ação no ambiente. Retorna uma tupla (observation, reward, done, truncation, info).
            - reset(): Reinicia o ambiente e retorna uma observação inicial.
    """

    def __init__(self, network_load, k_routes, number_of_slots, state_type: str = 'int') -> None:
        """
            Construtor da classe Enviroment.

            Parâmetros:
                network_load: Carga da rede em Erlangs.
                k_routes: Número de rotas alternativas para cada par origem-destino.
                number_of_slots: Número de slots por enlace.
                state_type: Tipo de estado do ambiente. Pode ser 'int' ou 'one-hot'.
        """

        self._number_of_slots = number_of_slots
        self._network_load = network_load
        self._k_routes = k_routes
        self._state_type = state_type

        # Cria a topologia de rede NSFNet
        self.network = iTwo(num_of_slots = number_of_slots)
        self._number_of_nodes = self.network.get_num_of_nodes()

        # Cria o caminho de nós de todas as rotas possíveis
        if type(self.network) != iTwo:
            self._allRoutes_paths = [[self.network.k_shortest_paths(origin, source, k_routes) for source in range(self.network.get_num_of_nodes())] for origin in range(self.network.get_num_of_nodes())]

            # Cria a lista de rotas possíveis em rotas
            self.allRoutes = self.generate_routes(self._allRoutes_paths)
            self.update_iRoutes(self.allRoutes)

        elif type(self.network) == iTwo:
            self._allRoutes_paths = [
                                    [0, 1, 2, 3],
                                    [1, 2, 5, 6],
                                    [2, 3, 4, 5],
                                    [4, 5, 6, 7]]
            
            self.allRoutes = [
                [self.route_by_path(self._allRoutes_paths[0], 0)],
                [self.route_by_path(self._allRoutes_paths[1], 1)],
                [self.route_by_path(self._allRoutes_paths[2], 2)],
                [self.route_by_path(self._allRoutes_paths[3], 3)],
            ]

            self.update_iRoutes(self.allRoutes)


        self._demands_class = DEMANDS_CLASS

        # Define o espaço de ações para a saída do algoritmo. Como nosso estado de ação é 0 (RSA) e 1 (SAR). Usamos o Discrete(2) para definir o espaço de ações.
        self.action_space = gym.spaces.Discrete(2)

        if self._state_type == 'one-hot':
            self.observation_space = gym.spaces.MultiBinary(self._number_of_nodes * 2, seed=42)
        elif self._state_type == 'int':
            self.observation_space = gym.spaces.Box(low=0, high=self._number_of_nodes - 1, shape=(2,), dtype=np.int32, seed=42)

        # Cria um mapa de codificação para os estados de origem e destino em one hot encoding
        self._source_destination_map = np.eye(self._number_of_nodes)

        self._reward_by_step = []
        self._reward_episode = None

    def route_by_path(self, path, id_route):

        # Retorna o índice dos links que compõem o uplink da rota
        uplinks = [self.network.spectrum_map[path[i], path[i + 1]] for i in range(len(path) - 1)]
        downlinks = []

        uplinks_spectrum = []
        for link_id in uplinks:
            uplinks_spectrum.append(self.network.all_optical_links[link_id])

        route = Route(node_path = path, route_index = id_route, uplink_path=uplinks, downlink_path=downlinks, number_of_slots=self._number_of_slots, uplinks_spectrums=uplinks_spectrum)

        return route


    def generate_routes(self, allRoutes_paths):
        """ Gera uma lista de classe Route para todas as rotas possíveis.

            Parâmetros:
                allRoutes_paths: Lista de caminhos dos nós para todas as rotas possíveis.

            Retorna:
                Uma lista de classe Route para todas as rotas possíveis.
        """

        routes = []
        route_id = 0

        links_mapping = self.network.spectrum_map

        for source, routes_for_source_path in enumerate(allRoutes_paths):

            for destination, routes_for_destination_path in enumerate(routes_for_source_path):

                #print(f'Source: {str(source).zfill(2)} Destination: {str(destination).zfill(2)} | ', end='\n')

                if source == destination:
                    routes.append(None)
                    continue
                
                routes_by_OD = []
                for node_path in routes_for_destination_path:

                    # Retorna o índice dos links que compõem o uplink da rota
                    uplinks = [links_mapping[node_path[i], node_path[i + 1]] for i in range(len(node_path) - 1)]

                    if IS_BIDIRECTIONAL:

                        node_path_reversed = node_path[::-1]

                        downlinks = [links_mapping[node_path_reversed[i], node_path_reversed[i + 1]] for i in range(len(node_path_reversed) - 1)]
                    else:
                        downlinks = []

                    uplinks_spectrum = []
                    for link_id in uplinks:
                        uplinks_spectrum.append(self.network.all_optical_links[link_id])

                    route = Route(node_path = node_path, route_index = route_id, uplink_path=uplinks, downlink_path=downlinks, number_of_slots=self._number_of_slots, uplinks_spectrums=uplinks_spectrum)
                                  
                    routes_by_OD.append(route)
                    route_id += 1
                
                routes.append(routes_by_OD)


        return routes


    def update_iRoutes(self, allRoutes):
        """ Atualiza a lista de rotas interferentes para cada rota.

            Parâmetros:
                allRoutes: Lista de rotas possíveis.
        
        """

        for routes_by_OD in allRoutes:
            if routes_by_OD is None:
                continue

            for main_route in routes_by_OD:
                main_route.iRoutes = []

                currentRouteID = main_route._route_index

                # Percorre todas as Rotas
                for i_routes_by_OD in allRoutes:

                    if i_routes_by_OD is None:
                        continue

                    for i_route in i_routes_by_OD:
                            
                        if i_route._route_index == currentRouteID:
                            continue

                        # Se pelo menos um link for compartilhado pelas rotas há um conflito
                        for main_route_link_uplink in main_route._path_uplink:
                            for i_route_link_uplink in i_route._path_uplink:
                                if main_route_link_uplink == i_route_link_uplink:
                                    if i_route not in main_route.iRoutes:
                                        main_route.iRoutes.append(i_route)
                                        break


    def reset(self, seed = 42, options = None):
        """ Reinicia o ambiente as variáveis de estado e retorna uma observação inicial. A observação inicial depende do tipo de estado escolhido para o ambiente.

            Parâmetros:
                seed: Semente para geração de números aleatórios.
                options: Opções para o ambiente. Não utilizado.

            Retorna:
                Uma tupla (observation, info).
        
        """

        self._reward_by_step.append(self._reward_episode)
        self._reward_episode = 0

        self._random_generator = np.random.default_rng(seed)

        # Limpa os slots alocados na matriz de links
        self.network.clean_links()

        # Limpa a lista de demandas ativas
        self._list_of_demands = []

        # Tempo de simulação
        self._simulation_time = 0.0
        self._is_available_slots = False
        self._total_number_of_blocks = 0
        self._last_request = 0

        # Lista para armazenar as demandas ativas na rede
        self._list_of_demands = []

        return self.get_observation(), {}


    def get_source_destination(self):
        """ Sorteia uma origem e um destino diferentes. Esse processo é feito separado do método step para que possamos utilizar o par OD como estado da rede.
        
            Retorna:
                Uma tupla (source, destination).

        """
        if type(self.network) == iTwo:
            source = self._random_generator.integers(0, 4)
        else:
            source = self._random_generator.integers(0, self._number_of_nodes)

        destination = self._random_generator.integers(0, self._number_of_nodes)
        while source == destination:
            destination = self._random_generator.integers(0, self._number_of_nodes)

        return source, destination


    def get_observation(self):
        """ Retorna uma observação do ambiente. A observação depende do tipo de estado escolhido para o ambiente.
        
        """

        self.source, self.destination = self.get_source_destination()

        if self._state_type == 'int':
            return np.array([self.source, self.destination], dtype=np.int32)
        elif self._state_type == 'one-hot':
            return np.concatenate([self._source_destination_map[self.source],
                            self._source_destination_map[self.destination]])


    def step(self, action):
        """ Executa uma ação no ambiente e retorna uma tupla (observation, reward, done, truncation, info).

            Parâmetros:
                action: Ação a ser executada no ambiente.

            Retorna:
                Uma tupla (observation, reward, done, truncation, info).
        
        """

        source = self.source
        destination = self.destination

        self._isAvailableSlots = False

        demands_to_remove = []

        # Remove as demandas que expiraram
        for demand in self._list_of_demands:
            if demand.departure_time <= self._simulation_time:
                demands_to_remove.append(demand)

        for demand in demands_to_remove:
            self._list_of_demands.remove(demand)

            route = demand.route
            slots = demand.slots

            self.network.deallocate_slots(route, slots)

        # Adiciona um incremento ao tempo de simulacao conforme a carga da rede
        self._simulation_time += utils.exponencial(self._network_load, self._random_generator)

        # Sorteia uma demanda de slots entre [2,3,6]
        demand_class = self._demands_class[self._random_generator.integers(0, len(self._demands_class))]

        # Executa o First-Fit para o algoritmo RSA
        if action == 0:
            if type(self.network) == iTwo:
                route, slots = RSA_FirstFit.find_slots(self.allRoutes[source], demand_class)
            else:
                route, slots = RSA_FirstFit.find_slots(self.allRoutes[destination + source * self._number_of_nodes], demand_class)
        elif action == 1:
            if type(self.network) == iTwo:
                route, slots = SAR_FistFit.find_slots(self.allRoutes[source], demand_class)
            else:
                route, slots = SAR_FistFit.find_slots(self.allRoutes[destination + source * self._number_of_nodes], demand_class)
        elif action == 2:
            if type(self.network) == iTwo:
                route, slots = MSCL_combinado.find_slots(self.allRoutes[source], demand_class)
            else:
                route, slots = MSCL_combinado.find_slots(self.allRoutes[destination + source * self._number_of_nodes], demand_class)
        else:
            raise ValueError('Invalid action.')

        # Calcula o tempo de partida da demanda (tempo atual + tempo de duração da demanda)
        departure_time = self._simulation_time + utils.exponencial(MEAN_CALL_DURATION, self._random_generator)

        # Verifica se o conjunto de slots é igual a demanda de slots
        if len(slots) == demand_class:
            self._isAvailableSlots = True

            # Cria a demanda e seus atributos para serem utilizados na alocação
            demand = Demand(self._last_request, demand_class, slots, route, self._simulation_time, departure_time)

            self._list_of_demands.append(demand)

            self.network.allocate_slots(route, slots)
        else:
            self._isAvailableSlots = False
            self._total_number_of_blocks += 1

        self._last_request += 1

        reward_step = +1 if self._isAvailableSlots else -10000

        self._reward_episode += reward_step

        return self.get_observation(), reward_step, False, True if self._last_request >= 100000 else False, {}


    def plot_reward(self):
        import matplotlib.pyplot as plt
        plt.plot(self._reward_by_step, label='Reward by Step')

        # Média móvel dos últimos 50 episódios
        #mean_reward = np.convolve(self._reward_by_step, np.ones((50,))/50, mode='valid')
        #plt.plot(mean_reward, label='Mean of last 50 episodes')

        plt.ylabel('Reward')
        plt.xlabel('Episode')
        plt.title('Reward by Episode')
        plt.grid(True)
        plt.show()
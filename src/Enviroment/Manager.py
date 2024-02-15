


import numpy as np
import os
import gymnasium as gym

import Enviroment.utils as utils
from Enviroment.Settings import *

from Topology.NSFNet import NSFNet
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

    def __init__(self, network_load, k_routes, number_of_slots, enviroment_type: dict = None, data_folder: str = None) -> None:
        """
            Construtor da classe Enviroment.

            Parâmetros:
                network_load: Carga da rede (em Erlangs).
                k_routes: Número de rotas alternativas para cada par origem-destino.
                number_of_slots: Número de slots por enlace.
                state_type: Tipo de estado do ambiente. Segue a seguinte estrutura padrão:
                {
                    "Observation": "OD", (Representação da observação do ambiente. Pode ser 'OD', 'ODD-one-hot', 'availability-vector', 'all-network')
                    "Action": "RSA-SAR", (Representação da ação do ambiente. Pode ser 'RSA-SAR', 'Route', 'MSCL')
                    "Reward": "RL-defaut", (Representação da recompensa do ambiente. Pode ser 'PB', 'RL-defaut')
                    "StopCond": "Not", (Condição de parada do ambiente. Pode ser 'Not', 'MaxReq', '40kReqs', '10kReqs', 'First')
                    "StartCond": "5kReqs" (Condição de início do ambiente. Pode ser '5kReqs', 'Empty')
                }
        """

        self._number_of_slots = number_of_slots
        self._network_load = network_load
        self._k_routes = k_routes

        self._data_folder = data_folder

        if enviroment_type is None:
            self._enviroment_type = {
                "Observation": "OD",
                "Action": "RSA-SAR",
                "Reward": "RL-defaut",
                "StopCond": "Not",
                "StartCond": "Empty"
            }
        else:
            self._enviroment_type = enviroment_type

        # Cria a topologia de rede NSFNet
        self._network = NSFNet(num_of_slots = number_of_slots)
        self._number_of_nodes = self._network.get_num_of_nodes()

        # Cria o caminho de nós de todas as rotas possíveis
        self._allRoutes_paths = [[self._network.k_shortest_paths(origin, source, k_routes) for source in range(self._network.get_num_of_nodes())] for origin in range(self._network.get_num_of_nodes())]

        # Cria a lista de rotas possíveis em rotas
        self.allRoutes = self.generate_routes(self._allRoutes_paths)
        self.update_iRoutes(self.allRoutes)

        # Estabelece os tipos de demandas de slots possíveis para a requisição
        self._demands_class = DEMANDS_CLASS

        # Cria um mapa de codificação one-hot para os estados de origem e destino
        self._source_destination_map = np.eye(self._number_of_nodes)

        # Cria um mapa de codificação one-hot para as demandas de slots
        self.demand_class_map = np.eye(len(self._demands_class))

        # Inicializa uma estrutura para armazenar os dados para debug e análise do ambiente
        self._debug_data = {
            "reward_by_step": [],
            "reward_by_step_acum": [],
            "reward_episodes": [],
            "reward_episode": 0,
        }

        # ? Espaço reservado para o as configurações do ambiente em Gym ?

        # * Definindo o espaço de ações do ambiente conforme o tipo de ação escolhido
        if self._enviroment_type["Action"] == "RSA-SAR":
            # Define o espaço de ações para a saída do algoritmo. Como nosso estado de ação é 0 (RSA) e 1 (SAR). Usamos o Discrete(2) para definir o espaço de ações.
            self.action_space = gym.spaces.Discrete(2) 
        elif self._enviroment_type["Action"] == "Route":
            # Define o espaço de ações para a saída do algoritmo. Como nosso estado de ação é o índice da rota escolhida. Usamos o Discrete(self._k_routes) para definir o espaço de ações.
            self.action_space = gym.spaces.Discrete(self._k_routes)
        elif self._enviroment_type["Action"] == "MSCL":
            # Define o espaço de ações para a saída do algoritmo. 
            self.action_space = gym.spaces.Discrete(1)
        else:
            raise ValueError("Tipo de ação inválido. Escolha entre 'RSA-SAR' ou 'Route'.")
        
        
        # * Definindo o espaço de observação do ambiente conforme o tipo de estado escolhido
        if self._enviroment_type["Observation"] == "OD":
            # Define o espaço de observação para a entrada do algoritmo. O espaço 'OD' é um espaço de observação que representa o par de origem e destino em inteiro.
            self.observation_space = gym.spaces.Box(low=0, high=self._number_of_nodes - 1, shape=(2,), dtype=np.uint32, seed=42)
        elif self._enviroment_type["Observation"] == "ODD-one-hot":
            # Define o espaço de observação para a entrada do algoritmo. O espaço 'OD-one-hot' é um espaço de observação que representa o par de origem, destino e demanda em one-hot encoding.
            #self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self._number_of_nodes * 2 + len(self._demands_class),), dtype=np.uint8, seed=42)
            self.observation_space = gym.spaces.MultiBinary(self._number_of_nodes * 2 + len(self._demands_class))
        elif self._enviroment_type["Observation"] == "availability-vector":
            # self.observation_space = gym.spaces.Box(
            #     low=0, high=1, 
            #     shape=(self._number_of_nodes * 2 + 
            #            len(self._demands_class) + 
            #            self._k_routes * self._number_of_slots
            #     ,), 
            #     dtype=np.uint8, seed=42)
            self.observation_space = gym.spaces.MultiBinary(self._number_of_nodes * 2 + len(self._demands_class) + self._k_routes * self._number_of_slots)
        elif self._enviroment_type["Observation"] == "all-network":
            number_of_links = self._network.get_num_of_links()
            # self.observation_space = gym.spaces.Box(
            #     low=0, high=1,
            #     shape=(self._number_of_nodes * 2 + 
            #            len(self._demands_class) + 
            #            number_of_links * self._number_of_slots
            #     ,),
            #     dtype=np.uint8, seed=42)
            self.observation_space = gym.spaces.MultiBinary(self._number_of_nodes * 2 + len(self._demands_class) + number_of_links * self._number_of_slots)
        else:
            raise ValueError("Tipo de observação inválido. Escolha entre 'OD', 'ODD-one-hot', 'availability-vector' ou 'all-network'.")
        
        self.collect_data(True)

    def route_by_path(self, path, id_route):

        # Retorna o índice dos links que compõem o uplink da rota
        uplinks = [self._network.spectrum_map[path[i], path[i + 1]] for i in range(len(path) - 1)]
        downlinks = []

        uplinks_spectrum = []
        for link_id in uplinks:
            uplinks_spectrum.append(self._network.all_optical_links[link_id])

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

        links_mapping = self._network.spectrum_map

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
                        uplinks_spectrum.append(self._network.all_optical_links[link_id])

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


    def reset(self, seed = None, options = None):
        """ Reinicia o ambiente as variáveis de estado e retorna uma observação inicial. A observação inicial depende do tipo de estado escolhido para o ambiente.

            Parâmetros:
                seed: Semente para geração de números aleatórios.
                options: Opções para o ambiente. Não utilizado.

            Retorna:
                Uma tupla (observation, info).
        
        """

        self.seed = seed

        # Salva a recompensa do episódio atual e limpa a recompensa do episódio
        if self._debug_data["reward_episodes"] != []:
            self._debug_data["reward_episodes"].append(self._debug_data["reward_episode"])
        self._debug_data["reward_episode"] = 0

        # Reinicia o gerador de números aleatórios
        self._random_generator = np.random.default_rng(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

        # Limpa a lista de demandas ativas
        self._list_of_demands = []

        # Tempo de simulação
        self._simulation_time = 0.0
        self._is_available_slots = False
        self._total_number_of_blocks = 0
        self._start_blocking = 0
        self._reward_episode = 0
        self._last_request = 0

        # Limpa os slots alocados na matriz de links
        self._network.clean_links()

        if self._enviroment_type["StartCond"] == "5kReqs":
            # Executa 5 mil requisições para gerar um estado inicial usando o algoritmo RSA e SAR com o objetivo de preencher as rotas com demandas em igual proporção
            self.get_observation()
            for _ in range(5000):
                action = self._random_generator.integers(0, 2)
                self.step(action)

        # Limpa
        self._reward_episode = 0
        self._start_blocking = 0
    
        return self.get_observation(), {}


    def get_source_destination(self):
        """ Sorteia uma origem e um destino diferentes. Esse processo é feito separado do método step para que possamos utilizar o par OD como estado da rede.
        
            Retorna:
                Uma tupla (source, destination).

        """
        if type(self._network) == iTwo:
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

        # Sorteia a próxima demanda de slots entre [2,3,6] utilizada pela função step
        self.demand_class = self._demands_class[self._random_generator.integers(0, len(self._demands_class))]

        # Sorteia uma origem e um destino diferentes que serão utilizados pela função step para a alocação da demanda
        self.source, self.destination = self.get_source_destination()

        # Sorteio o próximo tempo do simulador e remove as demandas que expiraram com base no tempo atual
        self._simulation_time += utils.exponencial(self._network_load, self._random_generator)

        # Remove as demandas que expiraram
        demands_to_remove = []
        for demand in self._list_of_demands:
            if demand.departure_time <= self._simulation_time:
                demands_to_remove.append(demand)
        for demand in demands_to_remove:
            self._list_of_demands.remove(demand)
            route = demand.route
            slots = demand.slots
            self._network.deallocate_slots(route, slots)

        # * Definindo o espaço de observação do ambiente conforme o tipo de estado escolhido
        if self._enviroment_type["Observation"] == "OD":
            # Retorna o espaço de observação para a entrada do algoritmo. O espaço 'OD' é um espaço de observação que representa o par de origem e destino em inteiro.
            return np.array([self.source, self.destination], dtype=np.uint32)
        elif self._enviroment_type["Observation"] == "ODD-one-hot":
            # Retorna o espaço de observação para a entrada do algoritmo. O espaço 'OD-one-hot' é um espaço de observação que representa o par de origem, destino e demanda em one-hot encoding.
            return np.concatenate([self._source_destination_map[self.source],
                            self._source_destination_map[self.destination],
                            self.demand_class_map[self._demands_class.index(self.demand_class)]])
        elif self._enviroment_type["Observation"] == "availability-vector":
            
            routes_by_OD = self.allRoutes[self.destination + self.source * self._number_of_nodes]

            # Vetor de disponibilidade de slots para todas as rotas possíveis
            availability_vector = np.zeros((len(routes_by_OD), self._number_of_slots), dtype=np.bool_)
            for i, route in enumerate(routes_by_OD):
                availability_vector[i] = np.sum(route.get_uplinks(), axis=0, dtype=np.bool_)

            return np.concatenate(
                [self._source_destination_map[self.source],
                self._source_destination_map[self.destination],
                self.demand_class_map[self._demands_class.index(self.demand_class)],
                availability_vector.flatten()
                ])
        elif self._enviroment_type["Observation"] == "all-network":
            return np.concatenate(
                [self._source_destination_map[self.source],
                self._source_destination_map[self.destination],
                self.demand_class_map[self._demands_class.index(self.demand_class)],
                self._network.all_optical_links.flatten()
                ])
        else:
            raise ValueError("Tipo de observação inválido. Escolha entre 'OD', 'ODD-one-hot', 'availability-vector' ou 'all-network'.")


    def step(self, action):
        """ Executa uma ação no ambiente e retorna uma tupla (observation, reward, done, truncation, info).

            Parâmetros:
                action: Ação a ser executada no ambiente.

            Retorna:
                Uma tupla (observation, reward, done, truncation, info).
        
        """

        source = self.source
        destination = self.destination
        if source == destination:
            raise ValueError('Source and destination must be different.')

        self._isAvailableSlots = False

        if self._enviroment_type["Action"] == "RSA-SAR":

            # Executa o First-Fit para o algoritmo RSA
            if action == 0: # RSA
                route, slots = RSA_FirstFit.find_slots(self.allRoutes[destination + source * self._number_of_nodes], self.demand_class)
            elif action == 1: # SAR
                route, slots = SAR_FistFit.find_slots(self.allRoutes[destination + source * self._number_of_nodes], self.demand_class)
            else:
                raise ValueError('Invalid action.')

        elif self._enviroment_type["Action"] == "Route":
            route = self.allRoutes[destination + source * self._number_of_nodes][action]

            route, slots = RSA_FirstFit.find_slots([route], self.demand_class)

        elif self._enviroment_type["Action"] == "MSCL":
            route, slots = MSCL_combinado.find_slots(self.allRoutes[destination + source * self._number_of_nodes], self.demand_class)


        # Calcula o tempo de partida da demanda (tempo atual + tempo de duração da demanda)
        departure_time = self._simulation_time + utils.exponencial(MEAN_CALL_DURATION, self._random_generator)

        # Verifica se o conjunto de slots é igual a demanda de slots
        if len(slots) == self.demand_class:
            self._isAvailableSlots = True

            # Cria a demanda e seus atributos para serem utilizados na alocação
            demand = Demand(self._last_request, self.demand_class, slots, route, self._simulation_time, departure_time)

            self._list_of_demands.append(demand)

            self._network.allocate_slots(route, slots)
        else:
            self._isAvailableSlots = False
            self._total_number_of_blocks += 1
            self._start_blocking += 1

        self._last_request += 1

        #! Cálculo da recompensa

        if self._enviroment_type["Reward"] == "RL-defaut":
            reward_step = +1 if self._isAvailableSlots else -1
        elif self._enviroment_type["Reward"] == "PB":
            reward_step = +1 - self._total_number_of_blocks / self._last_request
        else:
            raise ValueError("Tipo de recompensa inválido. Escolha entre 'RL-defaut' ou 'PB'.")

        self._reward_episode += reward_step

        #! Condição de parada
        
        isDone = False
        isTruncated = False
        if self._enviroment_type["StopCond"] == "Not":
            isDone = False
        elif self._enviroment_type["StopCond"] == "MaxReq":
            if self._last_request == MAX_REQS:
                isTruncated = True
        elif self._enviroment_type["StopCond"] == "40kReqs":
            if self._last_request >= 40_000:
                isTruncated = True
        elif self._enviroment_type["StopCond"] == "10kReqs":
            if self._last_request >= 10_000:
                isTruncated = True
        elif self._enviroment_type["StopCond"] == "First":
            if self._isAvailableSlots:
                isDone = True
        else:
            raise ValueError("Tipo de condição de parada inválido. Escolha entre 'Not', 'MaxReq' ou '40kReqs'.")

        self._debug_data["reward_by_step"].append(reward_step)
        self._debug_data["reward_by_step_acum"].append(self._reward_episode)

        info = {
            'total_number_of_blocks': self._total_number_of_blocks,
            'simulation_time': self._simulation_time,
            'is_blocked': not self._isAvailableSlots,
            'last_request': self._last_request,
        }

        return self.get_observation(),  reward_step, isDone, isTruncated, info


    def plot_reward(self):
        import matplotlib.pyplot as plt
        plt.plot(self._debug_data["reward_episodes"], label='Reward by Episode')

        plt.ylabel('Reward')
        plt.xlabel('Episode')
        plt.title('Reward by Episode')
        plt.grid(True)
        plt.show()


    def collect_data(self, create_folder: bool = False):

        # Cria uma pasta para armazenar os dados do ambiente dentro da pasta 'logs'. A pasta deve ser criada com um nome único para cada execução do ambiente, exemplo: 'logs/{folder_name}_{qtd}'. Sendo qtd a quantidade de pasta com o mesmo nome.
        if create_folder:
            folders_name = os.listdir(f'../logs/')
            qtd = len([name for name in folders_name if self._data_folder in name]) + 1

            self.folder_name = f'../logs/{self._data_folder}_{str(qtd).zfill(3)}'

            os.makedirs(self.folder_name, exist_ok=True)

            # Salva a configuração do ambiente em um arquivo .json
            config_content = {
                "network_load": self._network_load,
                "k_routes": self._k_routes,
                "number_of_slots": self._number_of_slots,
                "enviroment_type": self._enviroment_type,
                "is_bidirectional": IS_BIDIRECTIONAL,
                "is_cost_type": IS_COST_TYPE,
                "demands_class": DEMANDS_CLASS,
                "max_reqs": MAX_REQS,
                "mean_call_duration": MEAN_CALL_DURATION,
                "env_type": self._enviroment_type
            }

            with open(f'{self.folder_name}/config.json', 'w') as file:
                file.write(str(config_content).replace('\'', '\"').replace('False', 'false').replace('True', 'true'))

            # Cria um header para o com os resultados por simulação
            with open(f'{self.folder_name}/results.csv', 'w') as file:
                file.write('Seed;Blocking_Probability;Requests;Reward\n')
        else:

            # Salva os resultados da simulação em um arquivo .csv
            with open(f'{self.folder_name}/results.csv', 'a') as file:
                file.write(f'{self.seed};{self._total_number_of_blocks / self._last_request};{self._last_request};{self._reward_episode}\n')

            # 


        return self._debug_data
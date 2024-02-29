
import time
import numpy as np

from Enviroment.Manager import Enviroment
from Enviroment.Settings import *

import torch as th
import torch.nn as nn
    
def single_load_DNN_PSO(load: float, k_routes: int, number_of_slots: int, alg_heuristic: int, seed: int = 42, env_type: dict = None, number_of_sim: int = 10, folder_name: str = None):
    """
    Simulates a single load in the network. 

    Parameters:
        - load: Network load in Erlangs.
        - k_routes: Number of routes to be considered.
        - number_of_slots: Number of slots in the network.
        - alg_heuristic: Heuristic to be used. 0 for RSA, 1 for SAR and 2 for MSCL.

    Returns:
        - (blocking_probability, simulation_time): A tuple containing the blocking probability and the simulation time.
    """
    # Avaliando a PB do modelo treinado
    enviroment_type_test = {
        "Observation": "ODD-one-hot",
        "Action": "RSA-SAR",
        "Reward": "RL-defaut",
        "StopCond": "40kReqs",
        "StartCond": "Empty"
    }

    # Cria o ambiente de simulação
    env = Enviroment(network_load = load, k_routes = k_routes, number_of_slots = number_of_slots, enviroment_type=enviroment_type_test, data_folder="DNN")

    input_size = env.observation_space.n
    output_size = env.action_space.n

    class DNN(nn.Module):
        def __init__(self, input_size, output_size):
            super(DNN, self).__init__()
            self.linear_layer_stack = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.LeakyReLU(inplace=True),
                nn.Linear(256, 128),
                nn.LeakyReLU(inplace=True),
                nn.Linear(128, output_size),
                nn.Softmax(dim=1) if output_size > 2 else nn.Sigmoid()
            )

        def forward(self, x):
            return self.linear_layer_stack(x)
            
    model = DNN(input_size, output_size)

    model.eval()

    with th.no_grad():

        # # Reseta o ambiente
        # state, _ = env.reset(seed)

        # # Inicia a contagem de tempo
        # start_time = time.time()

        # blocking = 0
        # for reqs in range(MAX_REQS):

        #     action = model(th.tensor(state, dtype=th.float32)).argmax().item()

        #     state, reward, done, _, info = env.step(action)

        #     if info["is_blocked"]:
        #         blocking += 1

        #     if done:
        #         break


        # return blocking / reqs, time.time() - start_time, reqs
    

        ## Retorna a PB para o modelo treinado
        print(f"Executa uma simulação com a carga {load}, número de slots {number_of_slots} e o algoritmo {'DNN'}...")

        np.random.seed(seed)
        seeds = np.random.randint(0, 100_000, number_of_sim, dtype=int)
        pbs = np.zeros(number_of_sim)
        reward = np.zeros(number_of_sim)

        # Inicia a contagem de tempo
        start_time = time.time()

        for i, seed in enumerate(seeds):

            print(f"Executando simulação {i+1} de {number_of_sim}")

            # Reseta o ambiente
            state, info = env.reset(int(seed))

            for reqs in range(MAX_REQS):

                action = model(th.tensor(state, dtype=th.float32)).argmax().item()

                state, _, done, trunk, info = env.step(action)

                if (done or trunk):
                    reward[i] = env._reward_episode
                    break

            #env.collect_data()

            pbs[i] = info['total_number_of_blocks'] / reqs

            print(f"Blocking Probability: {pbs[i]} | Reward: {reward[i]}")


        print(f"\nBlocking Probability: {np.mean(pbs)} | Min: {np.min(pbs)} | Max: {np.max(pbs)} | +- {np.std(pbs)}")
        print(f"Reward: {np.mean(reward)} | Min: {np.min(reward)} | Max: {np.max(reward)} | +- {np.std(reward)}")

        return np.mean(pbs), time.time() - start_time, np.mean(reward)

import time

from Enviroment.Manager import Enviroment
from Enviroment.Settings import *

from stable_baselines3 import DQN, A2C, PPO
    
def train_DQN(load: float, k_routes: int, number_of_slots: int, seed: int = 42) -> (float, float):
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

    # Cria o ambiente de simulação
    env = Enviroment(network_load = load, k_routes = k_routes, number_of_slots = number_of_slots, state_type="one-hot")

    # # Cria o modelo
    # model = DQN('MlpPolicy', env, verbose=2,
    #             batch_size=4096,
    #             learning_starts=1000,
    #             learning_rate=0.001,
    #             buffer_size=200000,
    #             exploration_fraction=0.1,
    #             exploration_final_eps=0.002,
    #             target_update_interval=3,
    #             gamma=0.99,
    #             tau=1.0)


    # # Treina o modelo
    # model.learn(total_timesteps=1000000, progress_bar=True)

    # Salva o modelo
    #model.save("DQN_RSA_SAR")

    # Gráica a recompensa
    #env.plot_reward()

    # Carrega o modelo
    model = DQN.load("A2C_RSA_SAR")

    ## Retorna a PB para o modelo treinado
    print("Testing the model...")

    # Reseta o ambiente
    state, _ = env.reset(seed)

    # Inicia a contagem de tempo
    start_time = time.time()

    blocking = 0
    for _ in range(MAX_REQS):

        alg_heuristic = model.predict(state)[0]

        _, reward, _, _, _ = env.step(alg_heuristic)

        if reward == -1:
            blocking += 1

    return blocking / MAX_REQS, time.time() - start_time


def train_A2C(load: float, k_routes: int, number_of_slots: int, seed: int = 42) -> (float, float):
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

    # Cria o ambiente de simulação
    env = Enviroment(network_load = load, k_routes = k_routes, number_of_slots = number_of_slots, state_type="one-hot")

    # # Cria o modelo
    # model = A2C('MlpPolicy', env, verbose=2)


    # # Treina o modelo
    # model.learn(total_timesteps=1000000, progress_bar=True)

    # # Salva o modelo
    # model.save("A2C_RSA_SAR_250E")

    # # Gráica a recompensa
    # env.plot_reward()

    model = A2C.load("A2C_RSA_SAR_250E")

    ## Retorna a PB para o modelo treinado
    print("Testing the model...")

    # Reseta o ambiente
    state, _ = env.reset(seed)

    # Inicia a contagem de tempo
    start_time = time.time()

    blocking = 0
    for _ in range(MAX_REQS):

        alg_heuristic = model.predict(state)[0]

        _, reward, _, _, _ = env.step(alg_heuristic)

        if reward != 1:
            blocking += 1

    return blocking / MAX_REQS, time.time() - start_time
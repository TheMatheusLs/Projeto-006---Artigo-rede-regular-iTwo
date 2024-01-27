
import time

from Enviroment.Manager import Enviroment
from Enviroment.Settings import *
import numpy as np

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
    env = Enviroment(network_load = load, k_routes = k_routes, number_of_slots = number_of_slots, state_type="multi_metrics")

    # Cria o modelo
    model = DQN('MlpPolicy', env, verbose=2,
                seed=42,
                learning_rate=0.00005,
                batch_size=2048,
                buffer_size=100000,
                learning_starts=10,
                train_freq=(2, "step"),
                gradient_steps=10,
                target_update_interval=5,
                exploration_fraction=0.98,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.002,
                tensorboard_log="./logs/",
    )

    # # Treina o modelo
    model.learn(total_timesteps=2_000_000, progress_bar=True)

    # Salva o modelo
    model.save("DQN_SAR_RSA")

    # Gráica a recompensa
    env.plot_reward()

    del model

    # Carrega o modelo
    model = DQN.load("DQN_SAR_RSA")

    ## Retorna a PB para o modelo treinado
    print("Testing the model...")

    # Reseta o ambiente
    state, _ = env.reset(seed)

    # Inicia a contagem de tempo
    start_time = time.time()

    actions = []

    blocking = 0
    for _ in range(MAX_REQS):

        alg_heuristic = model.predict(state)[0]
        actions.append(alg_heuristic)

        _, reward, _, _, info = env.step(alg_heuristic)

        if info["is_blocked"]:
            blocking += 1

    np.save("actions.npy", actions)

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
    env = Enviroment(network_load = load, k_routes = k_routes, number_of_slots = number_of_slots, state_type="multi_metrics")

    # Cria o modelo
    model = A2C('MlpPolicy', env, verbose=2, tensorboard_log="./logs/")

    # Treina o modelo
    model.learn(total_timesteps=2_000_000, progress_bar=True)

    # Salva o modelo
    model.save("A2C_RSA_SAR")

    # # Gráica a recompensa
    env.plot_reward()

    del model

    model = A2C.load("A2C_RSA_SAR")

    ## Retorna a PB para o modelo treinado
    print("Testing the model...")

    # Reseta o ambiente
    state, _ = env.reset(seed)

    # Inicia a contagem de tempo
    start_time = time.time()

    blocking = 0
    for _ in range(MAX_REQS):

        alg_heuristic = model.predict(state)[0]

        _, reward, _, _, info = env.step(alg_heuristic)

        if info["is_blocked"]:
            blocking += 1

    return blocking / MAX_REQS, time.time() - start_time

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
    env = Enviroment(network_load = load, k_routes = k_routes, number_of_slots = number_of_slots, state_type="RSA-SAR")

    # Cria o modelo
    model = DQN('MlpPolicy', env, verbose=2,
                seed=42,
                learning_rate=0.0005,
                batch_size=4096,
                buffer_size=100000,
                learning_starts=10,
                train_freq=(50, "step"),
                gradient_steps=10,
                target_update_interval=5,
                exploration_fraction=0.98,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.002,
                tensorboard_log="./logs/",
    )

    # # Treina o modelo
    model.learn(total_timesteps=1_000_000, progress_bar=True)

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
    for reqs in range(MAX_REQS):

        alg_heuristic = model.predict(state)[0]
        actions.append(alg_heuristic)

        state, reward, done, _, info = env.step(alg_heuristic)

        if info["is_blocked"]:
            blocking += 1

        if done: 
            break

    np.save("actions.npy", actions)

    return blocking / MAX_REQS, time.time() - start_time, reqs


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
    env = Enviroment(network_load = load, k_routes = k_routes, number_of_slots = number_of_slots, state_type="RSA-SAR")

    # Cria o modelo
    model = A2C(
            policy="MlpPolicy",
            env=env,
            learning_rate=0.0007,
            n_steps=5,
            gamma=0.99,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=42  # Escolha uma semente adequada para reprodução
    ) # By chatGPT

    # model = A2C(policy="MlpPolicy",env=env, ent_coef=0.01, gamma=0.99, learning_rate=0.0005, n_steps=2048, seed=42, verbose=1)
    

    # Treina o modelo
    model.learn(total_timesteps=1_000_000, progress_bar=True)

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
    for reqs in range(MAX_REQS):

        alg_heuristic = model.predict(state)[0]

        state, reward, done, _, info = env.step(alg_heuristic)

        if info["is_blocked"]:
            blocking += 1

        if done:
            break

    return blocking / reqs, time.time() - start_time, reqs


def train_HER(load: float, k_routes: int, number_of_slots: int, seed: int = 42) -> (float, float):
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

    from stable_baselines3 import HerReplayBuffer, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

    # Cria o ambiente de simulação
    env = Enviroment(network_load = load, k_routes = k_routes, number_of_slots = number_of_slots, state_type="RSA-SAR")

    model_class = DQN 

    goal_selection_strategy = "future"

    # Criar o agente com HER
    model = model_class(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        # Parameters for HER
        # replay_buffer_kwargs=dict(
        #     n_sampled_goal=4,
        #     goal_selection_strategy=goal_selection_strategy,
        # ),
        verbose=1,
    )

    # Treinar o agente
    model.learn(total_timesteps=100_000, progress_bar=True)


def train_PPO(load: float, k_routes: int, number_of_slots: int, seed: int = 42) -> (float, float):
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
    env = Enviroment(network_load = load, k_routes = k_routes, number_of_slots = number_of_slots, state_type="RSA-SAR")

    # Cria o modelo
    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     verbose=1,
    # )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=16,
        learning_rate=0.00001,
        batch_size=4048,
        ent_coef=0.01
    )

    # n_step
    # Lr
    # ent-coef


    # Treina o modelo
    model.learn(total_timesteps=1_000_000, progress_bar=True)

    # Salva o modelo
    model.save("PPO_RSA_SAR")

    # # Gráica a recompensa
    env.plot_reward()

    del model

    model = PPO.load("PPO_RSA_SAR")

    ## Retorna a PB para o modelo treinado
    print("Testing the model...")

    # Reseta o ambiente
    state, _ = env.reset(seed)

    # Inicia a contagem de tempo
    start_time = time.time()

    blocking = 0
    for reqs in range(MAX_REQS):

        alg_heuristic = model.predict(state)[0]

        state, reward, done, _, info = env.step(alg_heuristic)

        if info["is_blocked"]:
            blocking += 1

        if done:
            break

    return blocking / reqs, time.time() - start_time, reqs

import time
import numpy as np
from Enviroment.Manager import Enviroment
from Enviroment.Settings import MAX_REQS
    
def single_load(load: float, k_routes: int, number_of_slots: int, alg_heuristic: int, seed: int = 42, env_type: dict = None, number_of_sim: int = 10, folder_name: str = None) -> (float, float):
 

    # Cria o ambiente de simulação
    env = Enviroment(network_load = load, k_routes = k_routes, number_of_slots = number_of_slots, enviroment_type=env_type, data_folder=folder_name)

    ## Retorna a PB para o modelo treinado
    print(f"Executa uma simulação com a carga {load}, número de slots {number_of_slots} e o algoritmo {alg_heuristic}...")

    np.random.seed(seed)
    seeds = np.random.randint(0, 100_000, number_of_sim, dtype=int)
    pbs = np.zeros(number_of_sim)
    reward = np.zeros(number_of_sim)

    # Inicia a contagem de tempo
    start_time = time.time()

    for i, seed in enumerate(seeds):

        print(f"Executando simulação {i+1} de {number_of_sim}")

        # Reseta o ambiente
        _, info = env.reset(int(seed))

        for reqs in range(MAX_REQS):

            _, _, done, trunk, info = env.step(alg_heuristic)

            if (done or trunk):
                reward[i] = env._reward_episode
                break

        env.collect_data()

        pbs[i] = info['total_number_of_blocks'] / reqs

        print(f"Blocking Probability: {pbs[i]} | Reward: {reward[i]}")


    print(f"\nBlocking Probability: {np.mean(pbs)} | Min: {np.min(pbs)} | Max: {np.max(pbs)} | +- {np.std(pbs)}")
    print(f"Reward: {np.mean(reward)} | Min: {np.min(reward)} | Max: {np.max(reward)} | +- {np.std(reward)}")

    return np.mean(pbs), time.time() - start_time, np.mean(reward)
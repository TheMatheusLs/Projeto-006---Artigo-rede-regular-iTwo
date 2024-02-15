# from Simulations.train_SB3 import train_DQN, train_A2C, train_HER, train_PPO
# from Simulations.singleLoad import single_load
# from Enviroment.Settings import *

# # # Simulando o algoritmo RSA para as diversas cargas da rede
# # blocking_probabilitie, simulation_time, reqs = single_load(LOAD, K_ROUTES, NUMBER_OF_SLOTS, RSA_CODE, 42)

# # print("** RSA **")
# # print(f"Blocking Probability: {blocking_probabilitie}")
# # print(f"Simulation Time: {simulation_time}")
# # print(f"Requests: {reqs}")

# # # Simulando o algoritmo RSA para as diversas cargas da rede
# # blocking_probabilitie, simulation_time, reqs = single_load(LOAD, K_ROUTES, NUMBER_OF_SLOTS, SAR_CODE, 42)

# # print("** SAR **")
# # print(f"Blocking Probability: {blocking_probabilitie}")
# # print(f"Simulation Time: {simulation_time}")
# # print(f"Requests: {reqs}")

# # Treinando e simulando o algoritmo DQN
# blocking_probabilitie, simulation_time, reqs = train_PPO(LOAD, K_ROUTES, NUMBER_OF_SLOTS, 42)

# print("** Model: DQN **")
# print(f"Blocking Probability: {blocking_probabilitie}")
# print(f"Simulation Time: {simulation_time}")
# print(f"Requests: {reqs}")
import os

import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common import results_plotter
stable_baselines3.__version__ # printing out stable_baselines version used
import gymnasium as gym
import numpy as np
import time
from Simulations.train_SB3 import train_DQN, train_A2C, train_HER, train_PPO
from Enviroment.Settings import *
from Enviroment.Manager import Enviroment
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

env_type = {
    "Observation": "all-network",
    "Action": "RSA-SAR",
    "Reward": "RL-defaut",
    "StopCond": "40kReqs"
}

# Cria o ambiente de simulação
env = Enviroment(
    network_load=LOAD,
    k_routes=K_ROUTES,
    number_of_slots=NUMBER_OF_SLOTS,
    enviroment_type=env_type,
    data_folder="PPO",
)

LOG_PATH = env.folder_name

env = Monitor(env, LOG_PATH + '\\training\\')


class SaveDataCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(SaveDataCallback, self).__init__(verbose)
    

    def _on_step(self) -> bool:
        return True
    
    # Chama a função ao final do episódio
    def _on_rollout_end(self) -> None:
        # Log scalar value (here a random variable)
        env.collect_data()


saveCallback = SaveDataCallback()

# #Use deterministic actions for evaluation
# eval_callback = EvalCallback(env, best_model_save_path=LOG_PATH + "\\best_model\\",
#                                 log_path=LOG_PATH, eval_freq=10000,
#                                 deterministic=True, render=False, verbose=1, n_eval_episodes=1)

# here goes the arguments of the policy network to be used
policy_args = dict(net_arch=[2048, 512, 256, 128, 64]) # we use the elu activation function

agent = PPO(MlpPolicy, env, verbose=0, tensorboard_log=f"./{LOG_PATH}/tb/", policy_kwargs=policy_args, gamma=.95, learning_rate=1e-5)


a = agent.learn(total_timesteps=100_000, callback=saveCallback, progress_bar=True)


# Cria o ambiente de simulação
env = Enviroment(
    network_load=LOAD,
    k_routes=K_ROUTES,
    number_of_slots=NUMBER_OF_SLOTS,
    enviroment_type=env_type,
    data_folder="PPO",
)

model = PPO.load(f"{LOG_PATH}\\best_model\\best_model.zip")

## Retorna a PB para o modelo treinado
print("Testing the model...")

NUM_SIM = 10
np.random.seed(42)
seeds = np.random.randint(0, 100_000, NUM_SIM, dtype=int)
pbs = np.zeros(NUM_SIM)
reward = np.zeros(NUM_SIM)

# Inicia a contagem de tempo
start_time = time.time()

for i, seed in enumerate(seeds):

    print(f"Executando simulação {i+1} de {NUM_SIM}")

    # Reseta o ambiente
    state, info = env.reset(int(seed))

    for reqs in range(MAX_REQS):

        alg_heuristic = model.predict(observation=state, deterministic=True)[0]

        state, _, done, trunk, info = env.step(alg_heuristic)

        if (done or trunk) and reward[i] == 0:
            reward[i] = env._reward_episode

    pbs[i] = info['total_number_of_blocks'] / reqs

    print(f"Blocking Probability: {pbs[i]} | Reward: {reward[i]}")

print(f"\nBlocking Probability: {np.mean(pbs)} | Min: {np.min(pbs)} | Max: {np.max(pbs)} | +- {np.std(pbs)}")
print(f"Reward: {np.mean(reward)} | Min: {np.min(reward)} | Max: {np.max(reward)} | +- {np.std(reward)}")
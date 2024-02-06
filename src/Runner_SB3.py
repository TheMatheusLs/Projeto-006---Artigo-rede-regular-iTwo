from Simulations.train_SB3 import train_DQN, train_A2C, train_HER, train_PPO
from Simulations.singleLoad import single_load
from Enviroment.Settings import NUMBER_OF_SLOTS

RSA_CODE = 0
SAR_CODE = 1
MSCL_CODE = 2

LOAD = 250

K_ROUTES = 3


# Treinando e simulando o algoritmo DQN
blocking_probabilitie, simulation_time, reqs = train_PPO(LOAD, K_ROUTES, NUMBER_OF_SLOTS, 42)

print("** Model: DQN **")
print(f"Blocking Probability: {blocking_probabilitie}")
print(f"Simulation Time: {simulation_time}")
print(f"Requests: {reqs}")


# Simulando o algoritmo RSA para as diversas cargas da rede
blocking_probabilitie, simulation_time, reqs = single_load(LOAD, K_ROUTES, NUMBER_OF_SLOTS, RSA_CODE, 42)

print("** RSA **")
print(f"Blocking Probability: {blocking_probabilitie}")
print(f"Simulation Time: {simulation_time}")
print(f"Requests: {reqs}")

# Simulando o algoritmo RSA para as diversas cargas da rede
blocking_probabilitie, simulation_time, reqs = single_load(LOAD, K_ROUTES, NUMBER_OF_SLOTS, SAR_CODE, 42)

print("** SAR **")
print(f"Blocking Probability: {blocking_probabilitie}")
print(f"Simulation Time: {simulation_time}")
print(f"Requests: {reqs}")

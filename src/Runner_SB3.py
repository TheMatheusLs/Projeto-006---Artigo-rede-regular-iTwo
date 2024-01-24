from Simulations.train_SB3 import train_DQN, train_A2C
from Simulations.singleLoad import single_load

RSA_CODE = 0
SAR_CODE = 1
MSCL_CODE = 2

LOAD = 250

K_ROUTES = 3
NUMBER_OF_SLOTS = 128


# Treinando e simulando o algoritmo DQN
blocking_probabilitie, simulation_time = train_A2C(LOAD, K_ROUTES, NUMBER_OF_SLOTS, 42)

print("** Model: DQN **")
print(f"Blocking Probability: {blocking_probabilitie}")
print(f"Simulation Time: {simulation_time}")

# Simulando o algoritmo RSA para as diversas cargas da rede
blocking_probabilitie, simulation_time = single_load(LOAD, K_ROUTES, NUMBER_OF_SLOTS, RSA_CODE, 42)

print("** RSA **")
print(f"Blocking Probability: {blocking_probabilitie}")
print(f"Simulation Time: {simulation_time}")

# Simulando o algoritmo RSA para as diversas cargas da rede
blocking_probabilitie, simulation_time = single_load(LOAD, K_ROUTES, NUMBER_OF_SLOTS, SAR_CODE, 42)

print("** SAR **")
print(f"Blocking Probability: {blocking_probabilitie}")
print(f"Simulation Time: {simulation_time}")


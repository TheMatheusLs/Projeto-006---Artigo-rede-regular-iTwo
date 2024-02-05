from Simulations.singleLoad import single_load
from Enviroment.Settings import NUMBER_OF_SLOTS

RSA_CODE = 0
SAR_CODE = 1
MSCL_CODE = 2

LOAD = 300

K_ROUTES = 3

# Simulando o algoritmo RSA para as diversas cargas da rede
blocking_probabilitie, simulation_time = single_load(LOAD, K_ROUTES, NUMBER_OF_SLOTS, RSA_CODE, 42) # Pode testar com RSA_CODE, SAR_CODE

print("** RSA FF**")
print(f"Blocking Probability: {blocking_probabilitie:e}")
print(f"Simulation Time: {simulation_time}")

# Simulando o algoritmo RSA para as diversas cargas da rede
blocking_probabilitie, simulation_time = single_load(LOAD, K_ROUTES, NUMBER_OF_SLOTS, SAR_CODE, 42) # Pode testar com RSA_CODE, SAR_CODE

print(f"Blocking Probability: {blocking_probabilitie:e}")
print(f"Simulation Time: {simulation_time}")

# # Simulando o algoritmo RSA para as diversas cargas da rede
# blocking_probabilitie, simulation_time = single_load(LOAD, K_ROUTES, NUMBER_OF_SLOTS, MSCL_CODE, 42) # Pode testar com RSA_CODE, SAR_CODE

# print("** MSCL Combinado **")
# print(f"Blocking Probability: {blocking_probabilitie:e}")
# print(f"Simulation Time: {simulation_time}")
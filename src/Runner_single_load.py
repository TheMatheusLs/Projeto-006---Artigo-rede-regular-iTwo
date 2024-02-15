from Simulations.singleLoad import single_load
from Simulations.singleLoad_GA import single_load_GA
from Enviroment.Settings import *
import numpy as np

env_type = {
    "Observation": "OD",
    "Action": "RSA-SAR",
    "Reward": "RL-defaut",
    "StopCond": "40kReqs"
}

pb_RSA, time_RSA, reward_RSA = single_load(LOAD, K_ROUTES, NUMBER_OF_SLOTS, RSA_CODE, 42, env_type, 3, "RSA")

pb_SAR, time_SAR, reward_SAR = single_load(LOAD, K_ROUTES, NUMBER_OF_SLOTS, SAR_CODE, 42, env_type, 3, "SAR")


# for i in range(50):
#     # Simulando o algoritmo RSA para as diversas cargas da rede
#     blocking_probabilitie, simulation_time, reqs = single_load(LOAD, K_ROUTES, NUMBER_OF_SLOTS, RSA_CODE, None) # Pode testar com RSA_CODE, SAR_CODE

#     # print("** RSA FF**")
#     # print(f"Blocking Probability: {blocking_probabilitie:e}")
#     # print(f"Simulation Time: {simulation_time}")
#     # print(f"Requests: {reqs}")

#     mean.append(reqs)

# print(f"RSA Mean: {sum(mean)/len(mean)}")

# mean = []

# for i in range(50):
#     # Simulando o algoritmo RSA para as diversas cargas da rede
#     blocking_probabilitie, simulation_time, reqs = single_load(LOAD, K_ROUTES, NUMBER_OF_SLOTS, SAR_CODE, None) # Pode testar com RSA_CODE, SAR_CODE

#     # print("** RSA FF**")
#     # print(f"Blocking Probability: {blocking_probabilitie:e}")
#     # print(f"Simulation Time: {simulation_time}")
#     # print(f"Requests: {reqs}")

#     mean.append(reqs)

# print(f"RSA Mean: {sum(mean)/len(mean):e}")

# # mean = []

# chromossome = np.array([0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0,
#  0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1,        
#  1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,        
#  1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1,        
#  1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1,        
#  1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0,        
#  1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1,        
#  0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1,        
#  1, 1, 0, 1,]) #np.random.randint(2, size=196)  # Antiga

# # chromossome = np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
# #  0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1,    
# #  0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,    
# #  0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,    
# #  1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1,    
# #  1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1,    
# #  0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0,    
# #  0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1,    
# #  1, 0, 0, 1,])


# for i in range(50):
#     # Simulando o algoritmo RSA para as diversas cargas da rede
#     #blocking_probabilitie, simulation_time, reqs = single_load(LOAD, K_ROUTES, NUMBER_OF_SLOTS, SAR_CODE, None) # Pode testar com RSA_CODE, SAR_CODE

#     blocking_probabilitie, _, reqs = single_load_GA(LOAD, K_ROUTES, NUMBER_OF_SLOTS, chromossome, None)

#     # print("** RSA FF**")
#     # print(f"Blocking Probability: {blocking_probabilitie:e}")
#     # print(f"Simulation Time: {simulation_time}")
#     # print(f"Requests: {reqs}")

#     mean.append(reqs)

# print(f"RSA Mean: {sum(mean)/len(mean):e}")
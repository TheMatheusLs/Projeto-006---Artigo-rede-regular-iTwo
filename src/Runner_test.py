from Enviroment.Manager import Enviroment
from Enviroment.Settings import NUMBER_OF_SLOTS


env = Enviroment(network_load = 250, k_routes = 3, number_of_slots = NUMBER_OF_SLOTS, state_type="dict")

# Espaço de ações
print(f"Espaço de ações: {env.action_space}")

# Espaço de estados
print(f"Espaço de estados: {env.observation_space}")

# Ação aleatória
print(f"Ação aleatória: {env.action_space.sample()}")

# Estado aleatório
print(f"Estado aleatório: {env.observation_space.sample()}")

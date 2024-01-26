import matplotlib.pyplot as plt
from numpy import linspace

from Simulations.multi_load import multi_load

RSA_CODE = 0
SAR_CODE = 1
MSCL_CODE = 2

LOAD_MIN = 70
LOAD_MAX = 85
LOAD_POINTS = 5
SIM_POINTS = 1
K_ROUTES = 1
NUMBER_OF_SLOTS = 240


# Simulando o algoritmo RSA para as diversas cargas da rede
blocking_probabilities_RSA, simulation_times_RSA = multi_load(load_min = LOAD_MIN, load_max = LOAD_MAX, load_points = LOAD_POINTS, sim_points = SIM_POINTS, k_routes = K_ROUTES, number_of_slots = NUMBER_OF_SLOTS, alg_heuristic = RSA_CODE)

print("Blocking probabilities RSA: ", blocking_probabilities_RSA)
print("Simulation times RSA: ", simulation_times_RSA)

# Simulando o algoritmo SAR para as diversas cargas da rede
blocking_probabilities_SAR, simulation_times_SAR = multi_load(load_min = LOAD_MIN, load_max = LOAD_MAX, load_points = LOAD_POINTS, sim_points = SIM_POINTS, k_routes = K_ROUTES, number_of_slots = NUMBER_OF_SLOTS, alg_heuristic = SAR_CODE)

print("Blocking probabilities SAR: ", blocking_probabilities_SAR)
print("Simulation times SAR: ", simulation_times_SAR)

# Simulando o algoritmo MSCL para as diversas cargas da rede
blocking_probabilities_MSCL, simulation_times_MSCL = multi_load(load_min = LOAD_MIN, load_max = LOAD_MAX, load_points = LOAD_POINTS, sim_points = SIM_POINTS, k_routes = K_ROUTES, number_of_slots = NUMBER_OF_SLOTS, alg_heuristic = MSCL_CODE)

print("Blocking probabilities MSCL: ", blocking_probabilities_MSCL)
print("Simulation times MSCL: ", simulation_times_MSCL)

## Plotando o gráfico de PB e tempo de simulação

network_load = linspace(LOAD_MIN, LOAD_MAX, LOAD_POINTS)

fig = plt.figure(figsize=(10, 5))

ax1 = fig.add_subplot(121)
ax1.semilogy(network_load, blocking_probabilities_RSA, marker='o', label='RSA', linestyle='--')
ax1.semilogy(network_load, blocking_probabilities_SAR, marker='d', label='SAR', linestyle='-.')
ax1.semilogy(network_load, blocking_probabilities_MSCL, marker='v', label='MSCL', linestyle=':')
ax1.set_xlabel("Network Load (Erlang)")
ax1.set_ylabel("Blocking Probability")
ax1.legend()
ax1.set_ylim(1E-5, 1E-1)

ax1.grid()

ax2 = fig.add_subplot(122)
ax2.plot(network_load, simulation_times_RSA, marker='o', label='RSA', linestyle='--')
ax2.plot(network_load, simulation_times_SAR, marker='d', label='SAR', linestyle='-.')
ax2.plot(network_load, simulation_times_MSCL, marker='v', label='MSCL', linestyle=':')
ax2.set_xlabel("Network Load (Erlang)")
ax2.set_ylabel("Simulation Time (s)")
ax2.legend()
ax2.grid()

plt.show()



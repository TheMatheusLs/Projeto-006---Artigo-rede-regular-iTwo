import pygad

from Simulations.singleLoad_GA import single_load_GA
from Simulations.singleLoad import single_load
from Enviroment.Settings import NUMBER_OF_SLOTS, MAX_REQS
import numpy as np


RSA_CODE = 0
SAR_CODE = 1
MSCL_CODE = 2

LOAD = 300
K_ROUTES = 3


def fitness_function(ga_instance, solution, solution_idx):
    
    blocking_probabilitie, _, reqs = single_load_GA(LOAD, K_ROUTES, NUMBER_OF_SLOTS, solution, 42)

    return - blocking_probabilitie, reqs


# Simulando o algoritmo RSA para as diversas cargas da rede
blocking_probabilitie, simulation_time, reqs = single_load(LOAD, K_ROUTES, NUMBER_OF_SLOTS, RSA_CODE, 42) # Pode testar com RSA_CODE, SAR_CODE

print("** RSA FF **")
print(f"Blocking Probability: {blocking_probabilitie:e}")
print(f"Simulation Time: {simulation_time}")
print(f"Number of blocked calls: {blocking_probabilitie * reqs}")
print(f"Number of requests: {reqs}")

# Simulando o algoritmo RSA para as diversas cargas da rede
blocking_probabilitie, simulation_time, reqs = single_load(LOAD, K_ROUTES, NUMBER_OF_SLOTS, SAR_CODE, 42) # Pode testar com RSA_CODE, SAR_CODE

print("** SAR FF **")
print(f"Blocking Probability: {blocking_probabilitie:e}")
print(f"Simulation Time: {simulation_time}")
print(f"Number of blocked calls: {blocking_probabilitie * reqs}")
print(f"Number of requests: {reqs}")

chromossome = np.array([0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0,
 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1,        
 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,        
 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1,        
 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1,        
 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0,        
 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1,        
 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1,        
 1, 1, 0, 1,]) #np.random.randint(2, size=196)
blocking_probabilitie, reqs = fitness_function(None, chromossome, 0)

print("** Random**")
print(f"Blocking Probability: {-blocking_probabilitie:e}")
print(f"Number of blocked calls: {-blocking_probabilitie * reqs}")
print(f"Number of requests: {reqs}")



# num_generations = 500
# num_parents_mating = 16

# sol_per_pop = 50
# num_genes = 196

# keep_parents = 2

# gene_space = [0, 1]

# # Função de callback para monitorar a evolução
# def callback_generation(ga_instance):
#     generation = ga_instance.generations_completed

#     solution, solution_fitness, _ = ga_instance.best_solution()

#     avg_fitness = np.mean(ga_instance.last_generation_fitness)

#     print(f"** Generation = {generation}; Average fitness = {avg_fitness}; Best fitness = {solution_fitness}")
#     print(f"   Solution: {solution}")

#     return "Teste"



# ga_instance = pygad.GA(num_generations=num_generations,
#                        num_parents_mating=num_parents_mating,
#                        fitness_func=fitness_function,
#                        sol_per_pop=sol_per_pop,
#                        num_genes=num_genes,
#                        keep_parents=keep_parents,
#                        gene_space=gene_space,
#                        on_generation=callback_generation)

# ga_instance.run()

# ga_instance.plot_result()

# solution, solution_fitness, solution_idx = ga_instance.best_solution()
# print("Parameters of the best solution : {solution}".format(solution=solution))
# print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
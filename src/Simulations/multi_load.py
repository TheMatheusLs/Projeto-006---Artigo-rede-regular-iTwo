from Simulations.singleLoad import single_load
from numpy import linspace, random
    
def multi_load(load_min: float, load_max: float, load_points: int, sim_points:int, k_routes: int, number_of_slots: int, alg_heuristic: int) -> tuple:
    """ Executa simulações para vários valores de carga e retorna as probabilidades de bloqueio e os tempos de simulação.

        Parâmetros:
            -load_min (float): Carga mínima.
            -load_max (float): Carga máxima.
            -load_points (int): Número de pontos entre a carga mínima e a carga máxima. O mínimo e máximo são inclusos.
            -sim_points (int): Número de simulações para cada valor de carga.
            -k_routes (int): Número de rotas por par de nós.
            -number_of_slots (int): Número de slots por enlace.
            -alg_heuristic (int): Algoritmo de roteamento e alocação de espectro. 0 para o RSA, 1 para SAR.
    
    """

    loads = linspace(load_min, load_max, load_points)
    random.seed(42)
    seeds = random.randint(0, 1000000, sim_points)

    blocking_probabilities = []
    simulation_times = []

    for load in loads:

        blocking_probability = 0
        simulation_time = 0

        for seed in seeds:

            print("Load: ", load, " Seed: ", seed)

            sim_PB, sim_time = single_load(load = load, k_routes = k_routes, number_of_slots = number_of_slots, alg_heuristic = alg_heuristic, seed=seed)

            blocking_probability += sim_PB
            simulation_time += sim_time

        blocking_probabilities.append(blocking_probability/sim_points)
        simulation_times.append(simulation_time/sim_points)

    return blocking_probabilities, simulation_times
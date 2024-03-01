import numpy as np
from Enviroment.Settings import NUMBER_OF_SLOTS

def find_slots(routes: list, demands: int) -> np.array:
    """ Encontra os slots disponíveis para alocar a demanda usando o método RSA. O método RSA aloca a demanda na primeira rota que possua slots disponíveis seguindo a ordem em que as rotas aparecem na lista de rotas e usando o First Fit para alocar os slots.

        Parâmetros:
            -routes (list): Lista de rotas.
            -demands (int): Número de slots da demanda.

        Retorno:
            -route (Route): Rota onde foi possível alocar a demanda.
            -slots (list[int]): Lista de slots alocados.
    
    """

    # Percorre cada rota na ordem em que aparece na lista de rotas
    for route in routes:

        uplinks = route.get_uplinks()

        # Constroi o vetor de disponibilidade de slots para o uplink
        availability_vector_uplink = uplinks[0]
        for link in range(1, len(uplinks)):
            availability_vector_uplink = np.logical_or(availability_vector_uplink, uplinks[link])

        for slot in range(NUMBER_OF_SLOTS - demands + 1):
            is_available = True
            for j in range(demands):
                if availability_vector_uplink[slot + j]:
                    is_available = False
                    break
            if is_available:
                return route, list(range(slot, slot + demands))

    return None, []
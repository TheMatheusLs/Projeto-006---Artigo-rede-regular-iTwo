import numpy as np

# def find_slots(routes: list, demands: int):
#     number_of_slots = routes[0]._number_of_slots

#     for slot in range(number_of_slots - demands):
#         for route in routes:
#             uplinks = route.get_uplinks()

#             # Constroi o vetor de disponibilidade de slots para o uplink
#             availability_vector_uplink = np.any(uplinks, axis=0)

#             if not np.any(availability_vector_uplink[slot : slot + demands]):
#                 return route, list(range(slot, slot + demands))

#     return None, []



def find_slots(routes: list, demands: int) -> np.array:
    """ Encontra os slots disponíveis para alocar a demanda usando o método SAR. O método SAR aloca a demanda na rota que possua slots disponíveis no menor índice possível seguindo a ordem em que as rotas aparecem na lista de rotas.

        Parâmetros:
            -routes (list): Lista de rotas.
            -demands (int): Número de slots da demanda.

        Retorno:
            -route (Route): Rota onde foi possível alocar a demanda.
            -slots (list[int]): Lista de slots alocados.
    """

    number_of_slots = routes[0]._number_of_slots

    availability_vector_uplink_by_route = []

    # Percorre os slots de menor índice até o último slot possível para alocar a demanda
    for slot in range(number_of_slots - demands):
        # Percorre cada rota na ordem em que aparece na lista de rotas
        for index, route in enumerate(routes):

            if index == len(availability_vector_uplink_by_route):

                uplinks = route.get_uplinks()

                #Constroi o vetor de disponibilidade de slots para o uplink
                availability_vector_uplink = uplinks[0]
                for link in range(1, len(uplinks)):
                    availability_vector_uplink = np.logical_or(availability_vector_uplink, uplinks[link])

                availability_vector_uplink_by_route.append(availability_vector_uplink)

            #availability_vector_uplink = np.sum(route.get_uplinks(), axis=0, dtype=bool)

            is_available = True
            for j in range(demands):
                if availability_vector_uplink_by_route[index][slot + j]:
                    is_available = False
                    break
            if is_available:
                return route, list(range(slot, slot + demands))

    return None, []


# def find_slots(routes: list, demands: int) -> np.array:
#     """ Encontra os slots disponíveis para alocar a demanda usando o método SAR. O método SAR aloca a demanda na rota que possua slots disponíveis no menor índice possível seguindo a ordem em que as rotas aparecem na lista de rotas.

#         Parâmetros:
#             -routes (list): Lista de rotas.
#             -demands (int): Número de slots da demanda.

#         Retorno:
#             -route (Route): Rota onde foi possível alocar a demanda.
#             -slots (list[int]): Lista de slots alocados.
#     """

#     number_of_slots = routes[0]._number_of_slots

#     availability_vector_uplink_by_route = []

#     # Percorre os slots de menor índice até o último slot possível para alocar a demanda
#     for slot in range(number_of_slots - demands):
#         # Percorre cada rota na ordem em que aparece na lista de rotas
#         for route in routes:

#             # uplinks = route.get_uplinks()

#             # #Constroi o vetor de disponibilidade de slots para o uplink
#             # availability_vector_uplink = uplinks[0]
#             # for link in range(1, len(uplinks)):
#             #     availability_vector_uplink = np.logical_or(availability_vector_uplink, uplinks[link])

#             availability_vector_uplink = np.sum(route.get_uplinks(), axis=0, dtype=bool)

#             is_available = True
#             for j in range(demands):
#                 if availability_vector_uplink[slot + j]:
#                     is_available = False
#                     break
#             if is_available:
#                 return route, list(range(slot, slot + demands))

#     return None, []
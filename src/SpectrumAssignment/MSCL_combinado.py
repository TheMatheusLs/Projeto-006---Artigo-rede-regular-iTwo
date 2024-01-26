

import numpy as np
from typing import List, Tuple
from Enviroment.Settings import NUMBER_OF_SLOTS, DEMANDS_CLASS, CLASS_TAG_RSA_FF, CLASS_TAG_MSCL, CLASS_TAG_LF   
from Enviroment.route import Route

# MSCL Combinado
def find_slots(routes: list, demands: int) -> np.array:

    best_capacity_loss: float = float('inf')
    best_route = None
    best_slots = []

    # Percorre toda as rotas e retorna aquela rota e o conjunto de slots com a menor perda de capacidade entre todas as rotas
    for route in routes:

        capacity_loss, slots = find_capacity_loss(route, demands)

        if capacity_loss < best_capacity_loss:
            best_capacity_loss = capacity_loss
            best_route = route
            best_slots = slots
        

    return best_route, best_slots

def find_capacity_loss(route: Route, demands: int) -> (float, list[int]):

    main_route_links: list = route.get_uplinks()

    # Constroi o vetor de disponibilidade de slots para o uplink
    # main_route_availability_vector = main_route_links[0]
    # for link in range(1, len(main_route_links)):
    #     main_route_availability_vector = np.logical_or(main_route_availability_vector, main_route_links[link])

    main_route_availability_vector = np.sum(main_route_links, axis=0, dtype=bool)

    
    # Encontra todas as lacunas na rota principal
    all_apertures_in_main_route: list[(int, int)] = get_all_apertures(main_route_availability_vector)

    # Verifica se é possível alocar a demanda na rota principal
    is_possible_allocate: bool = False

    # Percorre todas as lacunas 
    aperture: (int, int)
    for aperture in all_apertures_in_main_route:
        # Verifica se ao menos uma lacuna tem o tamanho igual ou maior que a demanda
        if aperture[1] >= demands:
            is_possible_allocate = True
            break 
    
    # Se for possível alocar a demanda na rota principal retorna a capacidade infinita
    if is_possible_allocate == False:
        return float('inf'), []

    #*** Chegando aqui é possível alocar a demanda na rota principal ***#

    FF_slot_index: int = -1
    FF_capacity_loss = float('inf')

    LF_slot_index: int = -1
    LF_capacity_loss = float('inf')

    MSCL_slot_index: int = -1
    best_capacity_loss: float = float('inf')

    #** Calcula a capacidade ANTES da alocação da demanda

    capacity_before: float = find_capacity(route, all_apertures_in_main_route)
    
    #** Calcula a capacidade DEPOIS da alocação da demanda
    # Percorre todos os slots do espectro
    for slot_index in range(NUMBER_OF_SLOTS - demands + 1):

        # Delimita o início e o fim da demanda a ser alocada no espectro
        start_slot = slot_index
        final_slot = slot_index + demands - 1

        # Verifica as condições de parada para a alocação iniciando em 'slot_index'. São elas:
        # 1. As posições 'start_slot' e 'final_slot' não podem está ocupadas por outra demanda
        # 2. A posição 'final_slot' não pode ultrapassar o limite do espectro
        # 3. Todos os slots da demanda devem estar livres na rota principal
        stoped: bool = False
        if (main_route_availability_vector[start_slot] == True) or (main_route_availability_vector[final_slot] == True):
            continue

        for slot in range(start_slot+1, final_slot):
            if main_route_availability_vector[slot] == True:
                stoped = True

        if stoped == True:
            continue

        # * Chegando aqui, significa que a demanda pode ser alocada na rota principal

        # Atribui o valor de 'start_slot' para a variável 'best_FF_slot_index' caso ela ainda não tenha sido atribuída. Ou seja, é a primeira vez que um slot é encontrado pelo MSCL
        if FF_slot_index == -1:
            FF_slot_index = start_slot

        # Cria um vetor de slots que serão alocados na rota principal
        slots_req_fake = range(start_slot, final_slot + 1)

        #* Realiza uma alocação fake da demanda na rota principal
        route.fake_allocate(slots_req_fake)
        for slot in slots_req_fake:

            if main_route_availability_vector[slot] == True:
                raise Exception("Slot already allocated")

            main_route_availability_vector[slot] = True

        # Encontra todas as lacunas na rota principal após a alocação fake
        all_apertures_in_main_route = get_all_apertures(main_route_availability_vector)

        capacity_after: float = find_capacity(route, all_apertures_in_main_route)

        # Desaloca a requisição fake da rota principal para que a próxima iteração possa ser realizada
        route.fake_deallocate(slots_req_fake)
        for slot in slots_req_fake:

            if main_route_availability_vector[slot] == False:
                raise Exception("Slot already deallocated")

            main_route_availability_vector[slot] = False

        # Calcula a perda de capacidade para a alocação em 'slot_index'
        capacity_loss: float = capacity_before - capacity_after

        if capacity_loss < 0:
            raise Exception("Capacity loss is negative")
        
        # Se a perda de capacidade é do slot do First Fit, então armazena o valor
        if slot_index == FF_slot_index:
            FF_capacity_loss = capacity_loss

        # Como é a última iteração, então adiciona o valor de perda de capacidade do slot do Last Fit
        LF_capacity_loss = capacity_loss
        LF_slot_index = slot_index

        # ! Regras de alocação e desempate:
        # ! 1. Se o slot do First Fit tem a melhor perda de capacidade, então aloca a demanda nele
        # ! 2. Se o slot do Last Fit tem a melhor perda de capacidade, então aloca a demanda nele
        # ! 3. Caso contrário, aloca a demanda no slot do MSCL
        # ! 4. Os desempates do MSCL são quebrados pelo último slot disponível que teve a melhor perda de capacidade

        # ? A capacidade atual é a melhor até o momento ou há um entre duas perdas de capacidade iguais (menores que a perda de capacidade do FF e LF)?
        tie_condition = ((capacity_loss == capacity_loss) and (capacity_loss < FF_capacity_loss))

        if (capacity_loss < best_capacity_loss) or tie_condition:
            best_capacity_loss = capacity_loss
            MSCL_slot_index = slot_index

    # Verifica a compatibilidade entre as heurísticas
    class_tag: int = CLASS_TAG_MSCL
    if FF_slot_index == MSCL_slot_index:
        class_tag = CLASS_TAG_RSA_FF
    elif LF_slot_index == MSCL_slot_index:
        class_tag = CLASS_TAG_LF

    slots: list[int] = list(range(MSCL_slot_index, MSCL_slot_index + demands))

    return best_capacity_loss, slots


def find_capacity(route: Route, apertures_main_route: list[tuple[int, int]]) -> float:
    capacity_before: float = get_capacity_form(apertures_main_route)

    for iRoute in route.iRoutes:
        iRoute_availability_vector = np.sum(iRoute.get_uplinks(), axis=0, dtype=bool)
        capacity_before += get_capacity_form(get_all_apertures(iRoute_availability_vector))

    return capacity_before



# def find_capacity(route: Route, apertures_main_route: list[(int, int)]) -> float:

#     # Na rota principal
#     capacity_before: float = get_capacity_form(apertures_main_route)

#     # Nas rotas alternativas
#     alternative_routes: list[Route] = route.iRoutes
#     iRoute: Route
#     for iRoute in alternative_routes:
        
#         # Encontra o vetor de disponibilidade de slots para a rota iRoute
#         iRoute_links: list = iRoute.get_uplinks()

#         # iRoute_availability_vector = iRoute_links[0]
#         # for link in range(1, len(iRoute_links)):
#         #     iRoute_availability_vector = np.logical_or(iRoute_availability_vector, iRoute_links[link])

#         iRoute_availability_vector = np.sum(iRoute_links, axis=0, dtype=bool)

#         all_apertures_in_iRoute: list[(int, int)] = get_all_apertures(iRoute_availability_vector)


#         capacity_before += get_capacity_form(all_apertures_in_iRoute)

#     return capacity_before

def get_capacity_form(apertures: List[Tuple[int, int]]) -> float:
    capacity: float = 0

    for _, size in apertures:
        for demand in DEMANDS_CLASS:
            if demand > size:
                break
            # Equação do número de formas: (l - n + 1)
            capacity += (size - demand + 1)

    return capacity



def get_all_apertures(availability_vector: np.array) -> List[Tuple[int, int]]:

    all_apertures: List[Tuple[int, int]] = []
    size: int = 0

    for start_slot, is_available in enumerate(availability_vector):
        if not is_available:
            size += 1
        elif size > 0:
            all_apertures.append((start_slot - size, size))
            size = 0

    # Adiciona a última apertura, se houver, ao final do loop
    if size > 0:
        all_apertures.append((start_slot - size + 1, size))

    return all_apertures


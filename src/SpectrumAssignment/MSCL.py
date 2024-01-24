import numpy as np

def get_all_apertures(route, start_pos: int, end_pos: int) -> list[tuple]:

    availability_vector = route.get_links()[0].get_spectrum()[start_pos:end_pos + 1]
    for link in route.get_links()[1:]:
        availability_vector = np.logical_or(availability_vector, link.get_spectrum()[start_pos:end_pos + 1])

    apetures = []

    start = None
    for i, x in enumerate(availability_vector):
        if x == 0 and start is None:
            start = i
        elif x == 1 and start is not None:
            apetures.append((start, i - start))
            start = None
    if start is not None:
        apetures.append((start, len(availability_vector) - start))
    return apetures

class MSCLAlgorithm:
    def __init__(self, possible_demands: list[int], number_of_slots: int, PSO_table: np.ndarray):
        self.possible_demands = possible_demands
        self.number_of_slots = number_of_slots
        self.PSO_table = PSO_table


    def run(self, route: Route, demand: int) -> list[int]:  

        # Encontra todas as lacunas na rota principal começando na posição '0' e terminando na posição 'number_of_slots - 1'
        all_apertures_in_main_route = get_all_apertures(route, 0, self.number_of_slots - 1)

        # Verifica se é possível alocar a demanda na rota principal
        is_possible_allocate = False

        for index_slot, size in all_apertures_in_main_route:
            if size >= demand:
                is_possible_allocate = True
                break      

        # Se não for possível alocar a demanda na rota principal, retorna uma lista vazia
        if not is_possible_allocate:
            return [], -1

        # Inicializa a variável que armazena o melhor slot para alocar a demanda e seu respectivo valor de perda de capacidade
        best_capacity_loss = float("inf")
        best_MSCL_slot_index = -1
        best_FF_capacity_loss = float("inf")
        best_FF_slot_index = -1
        best_LF_capacity_loss = float("inf")
        best_LF_slot_index = -1

        # *** Calcula a perda de capacidade ANTES da alocação - Rota principal ***
        capacity_loss_before = 0

        # Percorre todas as lacunas da rota principal
        for start_aperture_pos, size_aperture in all_apertures_in_main_route:

            # Cálculo da perda de capacidade segundo a equação do número de formas
            for possible_demand in self.possible_demands:
                if possible_demand > size_aperture:
                    break

                capacity_loss_before += (size_aperture - possible_demand + 1)
            
        # ** Calcula a perda de capacidade ANTES da alocação - Rotas alternativas ***
        for alternative_route in route.get_iRoutes():
            
            # Encontra todas as lacunas na rota alternativa começando na posição '0' e terminando na posição 'number_of_slots - 1'
            all_apertures_in_alternative_route = get_all_apertures(alternative_route, 0, self.number_of_slots - 1)

            # Percorre todas as lacunas da rota alternativa
            for start_aperture_pos, size_aperture in all_apertures_in_alternative_route:

                # Cálculo da perda de capacidade segundo a equação do número de formas
                for possible_demand in self.possible_demands:
                    if possible_demand > size_aperture:
                        break

                    capacity_loss_before += (size_aperture - possible_demand + 1)

        # *** Calcula a perda de capacidade DEPOIS da alocação ***

        # Percorre todos os slots do espectro
        for slot_index in range(self.number_of_slots - demand + 1):

            # Delimita o início e o fim da demanda a ser alocada no espectro
            start_slot = slot_index
            final_slot = slot_index + demand - 1

            # Verifica as condições de parada para a alocação iniciando em 'slot_index'. São elas:
            # 1. As posições 'start_slot' e 'final_slot' não podem está ocupadas por outra demanda
            # 2. A posição 'final_slot' não pode ultrapassar o limite do espectro
            # 3. Todos os slots da demanda devem estar livres na rota principal
            if not route.is_available(start_slot, final_slot):
                continue
            
            # * Chegando aqui, significa que a demanda pode ser alocada na rota principal

            # Atribui o valor de 'start_slot' para a variável 'best_FF_slot_index' caso ela ainda não tenha sido atribuída. Ou seja, é a primeira vez que um slot é encontrado pelo MSCL
            if best_FF_slot_index == -1:
                best_FF_slot_index = start_slot

            # Cria um vetor de slots que serão alocados na rota principal
            slots_req_fake = np.arange(start_slot, final_slot + 1)

            # Realiza a alocação da requisição fake na rota principal
            route.allocate(slots_req_fake)

            # ** Calcula a perda de capacidade DEPOIS da alocação - Rota principal **
            capacity_loss_after = 0

            # Encontra as novas lacunas para a rota principal
            all_apertures_in_main_route_after = get_all_apertures(route, 0, self.number_of_slots - 1)

            # Percorre todas as lacunas da rota principal - Cálculo da perda de capacidade segundo a equação do número de formas
            for start_aperture_pos, size_aperture in all_apertures_in_main_route_after:
                for possible_demand in self.possible_demands:
                    if possible_demand > size_aperture:
                        break
                    capacity_loss_after += (size_aperture - possible_demand + 1)

            # ** Calcula a perda de capacidade DEPOIS da alocação - Rotas alternativas **

            # Percorre todas as rotas alternativas
            for alternative_route in route.get_iRoutes():
                all_apertures_in_alternative_route_after = get_all_apertures(alternative_route, 0, self.number_of_slots - 1)

                # Percorre todas as lacunas da rota alternativa - Cálculo da perda de capacidade segundo a equação do número de formas
                for start_aperture_pos, size_aperture in all_apertures_in_alternative_route_after:
                    for possible_demand in self.possible_demands:
                        if possible_demand > size_aperture:
                            break
                        capacity_loss_after += (size_aperture - possible_demand + 1)

            # Desaloca a requisição fake da rota principal para que a próxima iteração possa ser realizada
            route.deallocate(slots_req_fake) 

            # Calcula a perda de capacidade total
            total_capacity_loss = capacity_loss_before - capacity_loss_after

            # Verifica se a perda de capacidade para o slot 'indexSlot' é menor que zero e lança uma exceção. Busca por bugs
            if total_capacity_loss < 0:
                raise Exception("Erro na capacidade. Valor negativo.")
            
            # ? Se a perda de capacidade é do slot do First Fit, então armazena o valor
            if slot_index == best_FF_slot_index:
                best_FF_capacity_loss = total_capacity_loss

            # ! Como é a última iteração, então adiciona o valor de perda de capacidade do slot do Last Fit
            best_LF_capacity_loss = total_capacity_loss
            best_LF_slot_index = slot_index

            # ! Regras de alocação e desempate:
            # ! 1. Se o slot do First Fit tem a melhor perda de capacidade, então aloca a demanda nele
            # ! 2. Se o slot do Last Fit tem a melhor perda de capacidade, então aloca a demanda nele
            # ! 3. Caso contrário, aloca a demanda no slot do MSCL
            # ! 4. Os desempates do MSCL são quebrados pelo último slot disponível que teve a melhor perda de capacidade



            # ? A capacidade atual é a melhor até o momento ou há um entre duas perdas de capacidade iguais (menores que a perda de capacidade do FF e LF)?
            tie_condition = ((total_capacity_loss == best_capacity_loss) and (total_capacity_loss < best_FF_capacity_loss))

            if (total_capacity_loss < best_capacity_loss) or tie_condition:
                best_capacity_loss = total_capacity_loss
                best_MSCL_slot_index = slot_index

            # # Se a perda de capacidade para o slot 'indexSlot' é menor que a melhor perda de capacidade e atualiza a melhor perda de capacidade. Os casos de igualdade são tratados pegando o último slot que teve a melhor perda de capacidade, exceto quando o slot do First Fit tem a melhor perda de capacidade.
            # if (total_capacity_loss < best_capacity_loss) or (total_capacity_loss == best_capacity_loss and total_capacity_loss < best_FF_capacity_loss):
            #     best_capacity_loss = total_capacity_loss
            #     best_MSCL_slot_index = slot_index

        tag = 1 # 1 = MSCL
        if best_MSCL_slot_index == best_FF_slot_index:
            tag = 0 # 0 = FF
        elif best_MSCL_slot_index == best_LF_slot_index:
            tag = 2 # 2 = LF

        # Cria o vetor de slots que serão alocados na rota principal
        slots = np.arange(best_MSCL_slot_index, best_MSCL_slot_index + demand)

        return slots, tag

    
    def find_metrics(self, route: Route, demand: int) -> list[int]:
        # cpBefore_r1: 445
        # cpBefore_r2: 445
        # cpBefore_r3: 445
        # cp_FF_After_r1: 439
        # cp_FF_After_r2: 432
        # cp_FF_After_r3: 445
        # cp_LF_After_r1: 442
        # cp_LF_After_r2: 435
        # cp_LF_After_r3: 445
        # ocupationBefore_r1: 236
        # ocupationBefore_r2: 239
        # ocupationBefore_r3: 239
        # slotFF: 238
        # slotLF: 238
        # cpBefore: 1227
        # cp_FF_After: 1221
        # cp_LF_After: 1209
        # FF_CapacityLoss: 62
        # LF_CapacityLoss: 62  

        # Encontra todas as lacunas na rota principal começando na posição '0' e terminando na posição 'number_of_slots - 1'
        all_apertures_in_main_route = get_all_apertures(route, 0, self.number_of_slots - 1)

        # Verifica se é possível alocar a demanda na rota principal
        is_possible_allocate = False
        # aperture_index = 0

        for index_slot, size in all_apertures_in_main_route:
            if size >= demand:
                is_possible_allocate = True
                break      

        # Se não for possível alocar a demanda na rota principal, retorna uma lista vazia
        if not is_possible_allocate:
            return None
        
        # Variáveis de retorno
        cpsBefore = []
        cps_FF_After = []
        cps_LF_After = []
        ocupationBefore = []
        slotFF = -1
        slotLF = -1

        # *** Calcula a perda de capacidade ANTES da alocação - Rota principal ***
        capacity_route = 0
        # Percorre todas as lacunas da rota principal
        for start_aperture_pos, size_aperture in all_apertures_in_main_route:

            # Cálculo da perda de capacidade segundo a equação do número de formas
            for possible_demand in self.possible_demands:
                if possible_demand > size_aperture:
                    break

                capacity_route += (size_aperture - possible_demand + 1)

        # Armazena a capacidade da rota principal antes da alocação
        cpsBefore.append(capacity_route)

        # Encontrando a ocupação da rota principal antes da alocação
        availability_vector = route.get_links()[0].get_spectrum()
        for link in route.get_links()[1:]:
            availability_vector = np.logical_or(availability_vector, link.get_spectrum())

        ocupationBefore.append(availability_vector.sum())

        # ** Calcula a perda de capacidade ANTES da alocação - Rotas alternativas ***
        for alternative_route in route.get_iRoutes():
            
            # Encontra todas as lacunas na rota alternativa começando na posição '0' e terminando na posição 'number_of_slots - 1'
            all_apertures_in_alternative_route = get_all_apertures(alternative_route, 0, self.number_of_slots - 1)

            capacity_route = 0
            # Percorre todas as lacunas da rota alternativa
            for start_aperture_pos, size_aperture in all_apertures_in_alternative_route:

                # Cálculo da perda de capacidade segundo a equação do número de formas
                for possible_demand in self.possible_demands:
                    if possible_demand > size_aperture:
                        break

                    capacity_route += (size_aperture - possible_demand + 1)
            cpsBefore.append(capacity_route)

            # Encontrando a ocupação da rota principal antes da alocação
            availability_vector = alternative_route.get_links()[0].get_spectrum()
            for link in alternative_route.get_links()[1:]:
                availability_vector = np.logical_or(availability_vector, link.get_spectrum())

            ocupationBefore.append(availability_vector.sum())


        # *** Calcula a capacidade do FF DEPOIS da alocação ***

        # Percorre todos os slots do espectro
        for slot_index in range(self.number_of_slots - demand + 1):

            # Delimita o início e o fim da demanda a ser alocada no espectro
            start_slot = slot_index
            final_slot = slot_index + demand - 1

            # Verifica as condições de parada para a alocação iniciando em 'slot_index'. São elas:
            # 1. As posições 'start_slot' e 'final_slot' não podem está ocupadas por outra demanda
            # 2. A posição 'final_slot' não pode ultrapassar o limite do espectro
            # 3. Todos os slots da demanda devem estar livres na rota principal
            if not route.is_available(start_slot, final_slot):
                continue
            
            # Chegando aqui, significa que a demanda pode ser alocada na rota principal

            # Atribui o valor de 'start_slot' para a variável 'best_FF_slot_index' caso ela ainda não tenha sido atribuída. Ou seja, é a primeira vez que um slot é encontrado pelo MSCL
            if slotFF == -1:
                slotFF = start_slot

            # Cria um vetor de slots que serão alocados na rota principal
            slots_req_fake = np.arange(start_slot, final_slot + 1)

            # Realiza a alocação da requisição fake na rota principal
            route.allocate(slots_req_fake)

            # ** Calcula a perda de capacidade DEPOIS da alocação - Rota principal **
            capacity_loss_after = 0

            # Encontra as novas lacunas para a rota principal
            all_apertures_in_main_route_after = get_all_apertures(route, 0, self.number_of_slots - 1)

            capacity_route = 0
            # Percorre todas as lacunas da rota principal - Cálculo da perda de capacidade segundo a equação do número de formas
            for start_aperture_pos, size_aperture in all_apertures_in_main_route_after:
                for possible_demand in self.possible_demands:
                    if possible_demand > size_aperture:
                        break
                    capacity_route += (size_aperture - possible_demand + 1)
            cps_FF_After.append(capacity_route)

            # ** Calcula a perda de capacidade DEPOIS da alocação - Rotas alternativas **

            # Percorre todas as rotas alternativas
            for alternative_route in route.get_iRoutes():
                all_apertures_in_alternative_route_after = get_all_apertures(alternative_route, 0, self.number_of_slots - 1)

                capacity_route = 0
                # Percorre todas as lacunas da rota alternativa - Cálculo da perda de capacidade segundo a equação do número de formas
                for start_aperture_pos, size_aperture in all_apertures_in_alternative_route_after:
                    for possible_demand in self.possible_demands:
                        if possible_demand > size_aperture:
                            break
                        capacity_route += (size_aperture - possible_demand + 1)
                cps_FF_After.append(capacity_route)

            # Desaloca a requisição fake da rota principal para que a próxima iteração possa ser realizada
            route.deallocate(slots_req_fake) 

            # Como o FF já foi encontrado, interrompe o loop
            break

        # *** Calcula a capacidade do LF DEPOIS da alocação ***

        # Percorre todos os slots do espectro
        for slot_index in range(self.number_of_slots - demand, -1, -1):

            # Delimita o início e o fim da demanda a ser alocada no espectro
            start_slot = slot_index
            final_slot = slot_index + demand - 1

            # Verifica as condições de parada para a alocação iniciando em 'slot_index'. São elas:
            # 1. As posições 'start_slot' e 'final_slot' não podem está ocupadas por outra demanda
            # 2. A posição 'final_slot' não pode ultrapassar o limite do espectro
            # 3. Todos os slots da demanda devem estar livres na rota principal
            if not route.is_available(start_slot, final_slot):
                continue
            
            # Chegando aqui, significa que a demanda pode ser alocada na rota principal

            # Atribui o valor de 'start_slot' para a variável 'best_FF_slot_index' caso ela ainda não tenha sido atribuída. Ou seja, é a primeira vez que um slot é encontrado pelo MSCL
            if slotLF == -1:
                slotLF = start_slot

            # Cria um vetor de slots que serão alocados na rota principal
            slots_req_fake = np.arange(start_slot, final_slot + 1)

            # Realiza a alocação da requisição fake na rota principal
            route.allocate(slots_req_fake)

            # ** Calcula a perda de capacidade DEPOIS da alocação - Rota principal **
            capacity_loss_after = 0

            # Encontra as novas lacunas para a rota principal
            all_apertures_in_main_route_after = get_all_apertures(route, 0, self.number_of_slots - 1)

            capacity_route = 0
            # Percorre todas as lacunas da rota principal - Cálculo da perda de capacidade segundo a equação do número de formas
            for start_aperture_pos, size_aperture in all_apertures_in_main_route_after:
                for possible_demand in self.possible_demands:
                    if possible_demand > size_aperture:
                        break
                    capacity_route += (size_aperture - possible_demand + 1)
            cps_LF_After.append(capacity_route)

            # ** Calcula a perda de capacidade DEPOIS da alocação - Rotas alternativas **

            # Percorre todas as rotas alternativas
            for alternative_route in route.get_iRoutes():
                all_apertures_in_alternative_route_after = get_all_apertures(alternative_route, 0, self.number_of_slots - 1)

                capacity_route = 0
                # Percorre todas as lacunas da rota alternativa - Cálculo da perda de capacidade segundo a equação do número de formas
                for start_aperture_pos, size_aperture in all_apertures_in_alternative_route_after:
                    for possible_demand in self.possible_demands:
                        if possible_demand > size_aperture:
                            break
                        capacity_route += (size_aperture - possible_demand + 1)
                cps_LF_After.append(capacity_route)

            # Desaloca a requisição fake da rota principal para que a próxima iteração possa ser realizada
            route.deallocate(slots_req_fake)

            # Como o LF já foi encontrado, interrompe o loop
            break

        cp_before = (cpsBefore[0] + cpsBefore[1] + cpsBefore[2])
        cp_FF_after = (cps_FF_After[0] + cps_FF_After[1] + cps_FF_After[2])
        cp_LF_after = (cps_LF_After[0] + cps_LF_After[1] + cps_LF_After[2])

        return (cpsBefore[0] / 445, 
                cpsBefore[1] / 445,
                cpsBefore[2] / 445,
                cps_FF_After[0] / 445,
                cps_FF_After[1] / 445,
                cps_FF_After[2] / 445,
                cps_LF_After[0] / 445,
                cps_LF_After[1] / 445,
                cps_LF_After[2] / 445,
                ocupationBefore[0] / 240,
                ocupationBefore[1] / 240,
                ocupationBefore[2] / 240,
                slotFF / 240,
                slotLF / 240,
                cp_before / 1230,
                cp_FF_after / 1230,
                cp_LF_after / 1230,
                (cp_before - cp_FF_after) / 62,
                (cp_before - cp_LF_after) / 62
                )
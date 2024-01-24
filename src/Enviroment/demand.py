
class Demand:
    """ 
        Classe para representar uma demanda. A demanda é representada por um ID, uma classe de requisição, uma lista de slots, uma rota, um tempo de chegada e um tempo de partida.
    """
    def __init__(self, demand_ID: int, demand_class: int, slots: list[int], route, simulation_time: float, departure_time: float):
        """ 
            Construtor da classe Demand.
            
            Parâmetros:
                demand_ID: ID da demanda.
                demand_class: Classe da demanda.
                slots: Lista de slots alocados na demanda.
                route: Rota da demanda.
                simulation_time: Tempo de chegada da demanda.
                departure_time: Tempo de partida da demanda.
        """
        self.demand_ID = demand_ID
        self.demand_class = demand_class
        self.slots = slots
        self.route = route

        self.arrival_time = simulation_time
        self.departure_time = departure_time
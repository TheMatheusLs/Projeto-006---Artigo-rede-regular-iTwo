import numpy as np


class Route:
    """ Classe que representa uma rota de um par de nós.
    """

    def __init__(self, node_path: list[int], route_index: int, uplink_path: list[int], downlink_path: list[int], number_of_slots: int, uplinks_spectrums: list[np.ndarray] = None):
        """ Construtor da classe Route. A rota é definida por uma lista de nós e seus respectivos enlaces de uplink e downlink.

            Parâmetros:
                -node_path (list[int]): Lista de nós que compõem a rota.
                -route_index (int): Índice da rota.
                -uplink_path (list[int]): Lista de enlaces de uplink que compõem a rota.
                -downlink_path (list[int]): Lista de enlaces de downlink que compõem a rota.
                -number_of_slots (int): Número de slots da rota.
                -uplinks_spectrums (list[np.ndarray]): Lista de espectros de uplink da rota.

        """
        self._path_node = node_path
        self._path_uplink = uplink_path
        self._path_downlink = downlink_path
        self._route_index = route_index

        self.iRoutes = []

        self._number_of_slots = number_of_slots

        self.uplink_spectrums = uplinks_spectrums

    def get_uplinks(self):
        return self.uplink_spectrums
    
    def fake_allocate(self, slots: list[int]):
        """ Aloca uma demanda fake na rota principal.

            Parâmetros:
                -slots (list[int]): Lista de slots que serão alocados.
        """
        for link_spectrum in self.uplink_spectrums:
            for slot in slots:

                if link_spectrum[slot] == True:
                    raise Exception("Slot already allocated")

                link_spectrum[slot] = True

    def fake_deallocate(self, slots: list[int]):
        """ Desaloca uma demanda fake na rota principal.

            Parâmetros:
                -slots (list[int]): Lista de slots que serão desalocados.
        """
        for link_spectrum in self.uplink_spectrums:
            for slot in slots:

                if link_spectrum[slot] == False:
                    raise Exception("Slot already deallocated")

                link_spectrum[slot] = False 

    def __str__(self) -> str:
        return f"Route {self._route_index}: {self._path_node}"
    

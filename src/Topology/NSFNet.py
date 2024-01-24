# Cria a topologia de rede NSFNet usando o NetworkX

from Topology.Generic import Generic

class NSFNet(Generic):
    def __init__(self, num_of_slots: int):

        NUM_OF_NODES = 14

        # Adiciona os enlaces (representando as conexões) (source, destination, length)
        network_lenghts = [
            (0, 1, 1400.0),
            (0, 2, 800.0),
            (0, 3, 1200.0),
            (1, 2, 2000.0),
            (1, 7, 3400.0),
            (2, 5, 2400.0),
            (3, 4, 800.0),
            (3, 10, 2600.0),
            (4, 5, 1700.0),
            (4, 6, 800.0),
            (5, 9, 1300.0),
            (5, 13, 2300.0),
            (6, 7, 800.0),
            (7, 8, 800.0),
            (8, 9, 1100.0),
            (8, 11, 500.0),
            (8, 12, 600.0),
            (10, 11, 800.0),
            (10, 12, 1000.0),
            (11, 13, 500.0),
            (12, 13, 300.0)
        ]

        super().__init__(NUM_OF_NODES, num_of_slots, network_lenghts)
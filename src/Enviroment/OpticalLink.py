


class OpticalLink:

    def __init__(self, opticalLinkID, sourceNode, destinationNode, srlg, length) -> None:
        self.opticalLinkID = opticalLinkID
        self.sourceNode = sourceNode
        self.destinationNode = destinationNode
        self.srlg = srlg
        self.length = length
        self.link_state = True
        self.cost = 0

    def set_cost(self, cost):
        self.cost = cost

    def get_cost(self):
        return self.cost
    
    def set_link_state(self, isLinkWorking):
        self.link_state = isLinkWorking

    def is_link_working(self):
        return self.link_state
        
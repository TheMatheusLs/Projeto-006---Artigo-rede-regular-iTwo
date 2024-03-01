


class OpticalNode:
    def __init__(self, opticalSwitchID) -> None:
        self.opticalSwitchID = opticalSwitchID

        self.nodeWorking = True
        self.neighborNodes = []

    def set_node_state(self, isNodeWorking):
        self.nodeWorking = isNodeWorking

    def is_node_working(self):
        return self.nodeWorking
    
    def is_Equal(self, right):
        if self.opticalSwitchID == right.opticalSwitchID:
            return True
        return False

    def add_neighbor(self, neighbor):
        self.neighborNodes.append(neighbor)
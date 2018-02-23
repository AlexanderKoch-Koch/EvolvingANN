class InputNeuron:

    def __init__(self, name, graph):
        self.name = name
        self.output = 0
        self.graph = graph
        self.graph.add_node(self.name, color="r")

    def change_input(self, new_value):
        self.output = new_value
        self.graph.node[self.name]["color"] = "g" if self.output else "r"


from Neuron import Neuron


class InputNeuron:

    def __init__(self, name):
        self.name = name
        self.output = 0

    def change_input(self, new_value):
        self.output = new_value


from Neuron import Neuron


class InputNeuron(Neuron):

    def __init__(self):
        self.output = 0

    def change_input(self, new_value):
        self.output = new_value

    def compute(self):
        """do nothing"""
        self.output = self.output

    def read_inputs(self):
        """do nothing"""
        self.output = self.output

    def learn(self, reward):
        """do nothing"""
        self.output = self.output

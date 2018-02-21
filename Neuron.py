import random
from mathai import sigmoid


class Neuron:

    def __init__(self):
        self.inputs = []
        self.weights = []
        self.presynaptic_neurons = []
        self.output = 0
        """ tracks recent activity of synapse"""
        self.synapse_activity = []

    def add_input(self, presynaptic_neuron):
        """adds another input with standard input 0 and initializes random weight"""
        self.inputs.append(0)
        self.presynaptic_neurons.append(presynaptic_neuron)
        self.weights.append(random.randint(0, 100) / 100.0)  # random weight 0-1
        # self.weights.append((random.randint(0, 200) - 100) / 100.0)  # random weight -1-1
        self.synapse_activity.append(0)

    def read_inputs(self):
        for i in range(len(self.presynaptic_neurons)):
            self.inputs[i] = self.presynaptic_neurons[i].output

    def compute(self):
        """computes neuron outputs with current inputs and stores the result in output"""
        weighted_sum = 0
        for i in range(len(self.inputs)):
            weighted_sum += self.inputs[i] * self.weights[i]

        if weighted_sum > 1:
            self.output = 1
        else:
            self.output = 0

        return self.output

    def learn(self, reward):
        """positive reward reinforces behavior; negative reward results in forgetting"""
        if self.output == 1:
            for i in range(len(self.inputs)):
                if self.inputs[i] == 1:
                    self.weights[i] = sigmoid(self.weights[i] * (1 + reward))
        print(self.weights)
import random
from MathAI import sigmoid


class Neuron:

    def __init__(self, params, name="unknown"):
        self.p = params
        self.name = name
        self.inputs = []
        self.weights = []
        self.presynaptic_neurons = []
        self.output = 0
        self.synapse_activity = []  # tracks recent activity of synapse

    def add_input(self, presynaptic_neuron):
        """adds another input with standard input 0 and initializes random weight"""
        self.inputs.append(0)
        self.presynaptic_neurons.append(presynaptic_neuron)
        self.weights.append(1 / len(self.inputs))  # weight = 1 / len(self.inputs)
        self.synapse_activity.append(0)

    def read_inputs(self):
        """stores otput of presynaptic neurons in own variable"""
        for i in range(len(self.presynaptic_neurons)):
            self.inputs[i] = self.presynaptic_neurons[i].output

    def compute(self, firing_threshold):
        """computes neuron outputs and stores the result in self.output. Additionally, it returns the output"""
        firing_threshold *= len(self.inputs)
        # calculate weighted sum
        weighted_sum = 0
        for i in range(len(self.inputs)):
            if self.inputs[i] == 1:
                weighted_sum += self.weights[i]

        if weighted_sum > firing_threshold:
            # neuron fires
            self.output = 1
            # increase synapse activity for all currently active inputs
            for i in range(len(self.inputs)):
                # past activities are becoming more irrelevant
                self.synapse_activity[i] *= 0.97
                if self.inputs[i] == 1:
                    self.synapse_activity[i] += 1
        else:
            self.output = 0

        return self.output

    def learn(self, reward):
        """positive reward reinforces behavior; negative reward results in forgetting"""
        for i in range(len(self.inputs)):
            self.weights[i] += self.p["learning_rate"] * self.synapse_activity[i] * reward

        # print(self.weights)

    def reset(self):
        """reset recent synapse activity"""
        for i in range(len(self.inputs)):
            self.synapse_activity[i] = 0

import random
import math


class Neuron:

    def __init__(self):
        self.inputs = []
        self.weights = []

    def add_input(self):
        self.inputs.append(0)
        self.weights.append(random.randint(0, 100) / 100.0)  # random weight 0-1
        # self.weights.append((random.randint(0, 200) - 100) / 100.0)  # random weight -1-1

    def set_input(self, index, value):
        """inputs from 0 to 1"""
        self.inputs[index] = value

    def compute(self):
        weighted_sum = 0
        for i in range(len(self.inputs) - 1):
            weighted_sum += self.inputs[i] * self.weights[i]

        if weighted_sum > 1:
            return 1
        else:
            return 0

    def learn(self, reward):
        """positive reward reinforces behavior; negative reward results in forgetting"""
        for i in range(len(self.inputs) - 1):
            self.weights[i] *= (1 + math.fabs(self.inputs[i]) / len(self.inputs)) * (1 + reward)

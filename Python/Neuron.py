import random
from MathAI import sigmoid
import networkx


class Neuron:

    def __init__(self, params, name, graph):
        self.p = params
        self.name = name
        self.graph = graph
        self.inputs = []
        self.weights = []
        self.presynaptic_neurons = []
        self.output = 0
        self.synapse_activities = []  # tracks recent activity of synapse
        self.firing_threshold = self.p["initial_firing_threshold"]
        self.graph.add_node(self.name, color="r")

    def add_input(self, presynaptic_neuron):
        """adds another input with standard input 0 and initializes random weight"""
        self.inputs.append(0)
        self.presynaptic_neurons.append(presynaptic_neuron)
        self.weights.append(self.p["initial_weight"])
        num_inputs = len(self.presynaptic_neurons)
        self.graph.add_edge(self.presynaptic_neurons[num_inputs - 1].name, self.name, weight=self.p["initial_weight"] * 5)
        self.synapse_activities.append(0)

    def read_inputs(self):
        """stores otput of presynaptic neurons in own variable"""
        for i in range(len(self.presynaptic_neurons)):
            self.inputs[i] = self.presynaptic_neurons[i].output

    def compute(self, firing_threshold):
        self.firing_threshold = firing_threshold
        """computes neuron outputs and stores the result in self.output. Additionally, it returns the output"""
        #firing_threshold *= len(self.inputs)
        # calculate weighted sum
        weighted_sum = 0
        for i, input in enumerate(self.inputs):
            self.synapse_activities[i] *= self.p["synapse_activity_discount"]
            if input == 1:
                weighted_sum += self.weights[i]

        if weighted_sum > firing_threshold:
            # neuron fires
            self.output = 1
            # increase synapse activity for all currently active inputs
            for i, input in enumerate(self.inputs):
                if input == 1:
                    self.synapse_activities[i] += 1
        else:
            self.output = 0

        self.graph.node[self.name]["color"] = "g" if self.output else "r"
        return self.output

    def learn(self, reward):
        for i in range( len(self.inputs)):
            self.weights[i] += self.p["learning_rate"] * self.synapse_activities[i] * reward
            self.graph.edges[self.presynaptic_neurons[i].name, self.name]["weight"] = self.weights[i] * 5

    def reset(self):
        """reset recent synapse activity"""
        for i in range(len(self.synapse_activities)):
            self.synapse_activities[i] = 0

    def remove_synapse(self, synapse_index):
        self.graph.remove_edge(self.presynaptic_neurons[synapse_index].name, self.name)
        self.weights.pop(synapse_index)
        self.inputs.pop(synapse_index)
        self.presynaptic_neurons.pop(synapse_index)
        self.synapse_activities.pop(synapse_index)

from Neuron import Neuron
from InputNeuron import InputNeuron
import random
import networkx as nx
import matplotlib.pyplot as plt
from MathAI import round_randomly
import time


class Brain:
    """Class for managing Neurons"""

    def __init__(self, params):
        self.p = params
        self.neurons = []
        self.input_neurons = []
        self.firing_threshold = self.p["initial_firing_threshold"]
        self.graph = nx.DiGraph()
        self.node_colors = []
        plt.ion()
        plt.show()

        self.neuron_params = {
            "learning_rate": self.p["learning_rate"],
            "synapse_activity_discount": self.p["synapse_activity_discount"],
            "initial_firing_threshold": self.p["initial_firing_threshold"],
            "initial_weight": self.p["initial_weight"]
        }

        """Input neurons"""
        for i in range(self.p["num_inputs"]):
            self.input_neurons.append(InputNeuron(name="Input" + str(i), graph=self.graph))
            self.graph.node[self.input_neurons[i].name]["pos"] = (0, i)

        """Hidden neuron"""
        self.neurons.append(Neuron(self.neuron_params, "0", self.graph))
        self.graph.node[self.neurons[0].name]["pos"] = (1, 0)
        for input_neuron in range(self.p["num_inputs"]):
            self.neurons[0].add_input(self.input_neurons[input_neuron])

        """Output neurons"""
        self.output_neuron_index_range = range(1, self.p["num_outputs"] + 1)
        for output_neuron in self.output_neuron_index_range:
            name = "Output" + str(output_neuron)
            self.neurons.append(Neuron(self.neuron_params, name, self.graph))
            self.neurons[output_neuron].add_input(self.neurons[0])
            self.graph.node[self.neurons[output_neuron].name]["pos"] = (5, 0)

    def think(self, active_input_indexes):
        # activate inputs
        for i in active_input_indexes:
            self.input_neurons[i].change_input(1)

        # read inputs
        for neuron in self.neurons:
            neuron.read_inputs()

        for _ in range(self.p["think_steps"]):

            # compute
            num_neurons_fired = 0
            for i, neuron in enumerate(self.neurons):
                num_neurons_fired += neuron.compute(self.firing_threshold)

            self.draw_connectome()
            # deactivate input neurons
            for index, input_neuron in enumerate(self.input_neurons):
                input_neuron.change_input(0)

            # read inputs
            for neuron in self.neurons:
                neuron.read_inputs()

            # adjust firing_threshold
            self.firing_threshold += self.p["firing_threshold_factor"] * (num_neurons_fired / len(self.neurons) - self.p["target_firing_ratio"])

            # manage output
            for i in self.output_neuron_index_range:
                if self.neurons[i].output == 1:
                    return i - 1

            # regenerate
            for _ in range(round_randomly(self.p["regeneration_rate"])):
                self.neurons.append(Neuron(self.neuron_params, name=str(len(self.neurons)), graph=self.graph))
                x = random.randint(0, 300) / 100 + 1
                y = random.randint(0, 500) / 100
                self.graph.node[self.neurons[-1].name]["pos"] = (x, y)

            # reconnect
            for neuron in self.neurons:
                if sum(neuron.weights) > self.p["target_weight_sum"] + 1:
                    # remove synapse with smallest weight
                    neuron.remove_synapse(neuron.weights.index(min(neuron.weights)))
                elif sum(neuron.weights) < self.p["target_weight_sum"] - 1:
                    all_neurons = self.neurons + self.input_neurons
                    new_presynaptic_neuron = all_neurons[random.randint(0, len(all_neurons) - 1)]
                    # add if synapse doesn't yet exist
                    if new_presynaptic_neuron not in neuron.presynaptic_neurons:
                        neuron.add_input(new_presynaptic_neuron)

        # return None if no output was active
        return None

    def learn(self, reward):
        for neuron in self.neurons:
            neuron.learn(reward)

    def terminate_episode(self):
        for neuron in self.neurons:
            neuron.reset()

    def draw_connectome(self):
        plt.clf()

        pos = nx.get_node_attributes(self.graph, 'pos')
        node_color_dictionary = nx.get_node_attributes(self.graph, 'color')
        node_color_list = [v for v in node_color_dictionary.values()]

        weights = [self.graph[u][v]['weight'] for u, v in self.graph.edges]
        edge_colors = ["g" if x > 0 else "r" for x in weights]

        nx.draw(self.graph, pos=pos, node_color=node_color_list, width=weights, edge_color=edge_colors, with_labels=True, font_weight='bold')
        plt.pause(0.001)    # pause to let events be processed

        #time.sleep(0.4)

from Neuron import Neuron
from InputNeuron import InputNeuron
import random
import networkx as nx
import matplotlib.pyplot as plt

class Brain:
    """Class for managing Neurons"""

    def __init__(self, num_inputs, num_outputs):
        """initializing object variables"""
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.neurons = []   # array containing all neurons
        self.firing_threshold = 0.5  # initial firing threshold for hidden and output neurons
        self.learning_rate = 0.3
        self.regeneration_rate = 0.01    # adding 0.1 neurons per time step on average
        self.reconnection_rate = 0.1    # adding 0.5 synapse per time step to each neuron

        """initialize input neurons"""
        input_neuron_indexes = range(num_inputs)
        for i in input_neuron_indexes:
            self.neurons.append(InputNeuron("Input" + str(i)))

        """initialize first hidden neuron with all inputs"""
        self.normal_neuron_indexes = range(num_inputs, num_inputs + num_outputs)
        self.hidden_neurons_start_index = num_inputs
        self.neurons.append(Neuron(str(self.hidden_neurons_start_index), self.learning_rate))
        for input_neuron in input_neuron_indexes:
            self.neurons[self.hidden_neurons_start_index].add_input(self.neurons[input_neuron])

        """initialize Output neurons"""
        self.output_neuron_index_range = range(num_inputs + 1, num_inputs + num_outputs + 1)
        for i in self.output_neuron_index_range:
            name = "Output" + str(i - self.hidden_neurons_start_index)
            self.neurons.append(Neuron(name, self.learning_rate))
            self.neurons[i].add_input(self.neurons[self.hidden_neurons_start_index])

    def think(self, time_steps, active_input_indexes):
        """activate inputs"""
        for i in active_input_indexes:
            self.neurons[i].change_input(1)

        """think for given number of steps"""
        for _ in range(time_steps):
            """each neuron should save their inputs. This prohibits computation with newer inputs"""
            for neuron in self.normal_neuron_indexes:
                self.neurons[neuron].read_inputs()

            """"deactivate input neurons"""
            for i in range(self.num_inputs):
                self.neurons[i].change_input(0)

            """each neuron computes output from just saved inputs"""
            num_neurons_fired = 0
            for neuron in self.normal_neuron_indexes:
                num_neurons_fired += self.neurons[neuron].compute(self.firing_threshold)

            # adjust new firing_threshold
            self.firing_threshold += 0.02 * (num_neurons_fired / (len(self.neurons) - self.num_inputs) - 0.5)

            for i in self.output_neuron_index_range:
                if self.neurons[i].output == 1:
                    return i - self.output_neuron_index_range[0]

            # regenerate
            for _ in range(self.round_randomly(self.regeneration_rate)):
                self.neurons.append(Neuron(str(len(self.neurons)), self.learning_rate))

            # reconnect
            for neuron in self.normal_neuron_indexes:
                for _ in range(self.round_randomly(self.reconnection_rate)):
                    new_presynaptic_neuron = self.neurons[random.randint(0, len(self.neurons) - 1)]
                    if new_presynaptic_neuron not in self.neurons[neuron].presynaptic_neurons:
                        self.neurons[neuron].add_input(self.neurons[random.randint(0, len(self.neurons) - 1)])

        """return None if no output was active"""
        return None

    def learn(self, reward):
        for neuron in self.normal_neuron_indexes:
            self.neurons[neuron].learn(reward)

    def terminate_episode(self):
        for neuron in range(len(self.neurons)):
            self.neurons[neuron].reset()

    def round_randomly(self, x):
        return int(x) + (random.random() < x - int(x))

    def draw_connectome(self):
        """update connectome shown in pyplot"""
        G = nx.DiGraph()
        G.clear()
        # add all nodes/neurons as numbers
        for neuron in range(len(self.neurons)):
            G.add_node(self.neurons[neuron].name)

        # draw connections
        for neuron in range(self.hidden_neurons_start_index, len(self.neurons)):
            for i in range(len(self.neurons[neuron].presynaptic_neurons)):
                G.add_edge(self.neurons[neuron].presynaptic_neurons[i].name, self.neurons[neuron].name)

        plt.gcf().clear()
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.draw()
        plt.show(block=False)
        # plt.savefig("connectome.png") # uncomment to save drawing




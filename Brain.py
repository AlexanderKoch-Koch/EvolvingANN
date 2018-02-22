from Neuron import Neuron
from InputNeuron import InputNeuron
import random
import networkx as nx
import matplotlib.pyplot as plt
from MathAI import round_randomly


class Brain:
    """Class for managing Neurons"""

    def __init__(self, params):
        """initializing object variables"""
        self.p = params
        self.neurons = []   # array containing all neurons
        self.input_neurons = []
        self.firing_threshold = self.p["initial_firing_threshold"]

        self.neuron_params = {
            "learning_rate": self.p["learning_rate"]
        }

        """Input neurons"""
        for i in range(self.p["num_inputs"]):
            self.input_neurons.append(InputNeuron(name="Input" + str(i)))

        """Hidden neuron"""
        self.neurons.append(Neuron(self.neuron_params, "0"))
        for input_neuron in range(self.p["num_inputs"]):
            self.neurons[0].add_input(self.input_neurons[input_neuron])

        """Output neurons"""
        self.output_neuron_index_range = range(1, self.p["num_outputs"] + 1)
        for output_neuron in self.output_neuron_index_range:
            name = "Output" + str(output_neuron)
            self.neurons.append(Neuron(self.neuron_params, name))
            self.neurons[output_neuron].add_input(self.neurons[0])

    def think(self, think_steps, active_input_indexes):
        """activate inputs and think for given number of time steps"""
        for i in active_input_indexes:
            self.input_neurons[i].change_input(1)

        """"deactivate input neurons"""
        for i in range(len(self.input_neurons)):
            self.input_neurons[i].change_input(0)

        for _ in range(think_steps):
            print(str(len(self.neurons)) + "    firing_threshold:" + str(self.firing_threshold))
            """each neuron should save their inputs. This prohibits computation with newer inputs"""
            for neuron in range(len(self.neurons)):
                self.neurons[neuron].read_inputs()

            """each neuron computes output from just saved inputs"""
            num_neurons_fired = 0
            for neuron in range(len(self.neurons)):
                num_neurons_fired += self.neurons[neuron].compute(self.firing_threshold)

            # adjust new firing_threshold
            self.firing_threshold += 0.02 * (num_neurons_fired / len(self.neurons) - 0.5)

            for i in self.output_neuron_index_range:
                if self.neurons[i].output == 1:
                    return i - 1

            # regenerate
            for _ in range(round_randomly(self.p["regeneration_rate"])):
                self.neurons.append(Neuron(self.neuron_params, str(len(self.neurons))))

            # reconnect
            for neuron in range(len(self.neurons)):
                for _ in range(round_randomly(self.p["reconnection_rate"])):
                    new_presynaptic_neuron = self.neurons[random.randint(0, len(self.neurons) - 1)]
                    if new_presynaptic_neuron not in self.neurons[neuron].presynaptic_neurons:
                        self.neurons[neuron].add_input(self.neurons[random.randint(0, len(self.neurons) - 1)])

        """return None if no output was active"""
        return None

    def learn(self, reward):
        for neuron in range(len(self.neurons)):
            self.neurons[neuron].learn(reward)

    def terminate_episode(self):
        for neuron in range(len(self.neurons)):
            self.neurons[neuron].reset()

    def draw_connectome(self):
        G = nx.DiGraph()
        G.clear()
        # add nodes/neurons
        for neuron in range(len(self.neurons)):
            G.add_node(self.neurons[neuron].name)

        # draw connections
        for neuron in range(len(self.neurons)):
            for i in range(len(self.neurons[neuron].presynaptic_neurons)):
                weight = self.neurons[neuron].weights[i] * 10
                G.add_edge(self.neurons[neuron].presynaptic_neurons[i].name, self.neurons[neuron].name, weight=weight)

        plt.gcf().clear()
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.draw()
        plt.show(block=False)
        # plt.savefig("connectome.png") # uncomment to save drawing




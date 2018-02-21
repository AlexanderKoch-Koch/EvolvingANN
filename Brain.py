from Neuron import Neuron
from InputNeuron import InputNeuron
from OutputNeuron import OutputNeuron


class Brain:
    """Class for managing Neurons"""

    def __init__(self, num_inputs, num_outputs):
        """initializing neuron objects"""
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.output_callback_function = None
        self.neurons = []

        """initialize input neurons"""
        for i in range(num_inputs):
            self.neurons.append(InputNeuron())

        hidden_neuron_index = num_inputs
        """initialize first hidden neuron with all inputs"""
        self.neurons.append(Neuron())
        for i in range(num_inputs):
            self.neurons[hidden_neuron_index].add_input(self.neurons[i])

        """initialize Output neurons"""
        self.output_neuron_index_range = range(hidden_neuron_index + 1, hidden_neuron_index + 1 + num_outputs)
        for i in self.output_neuron_index_range:
            self.neurons.append(OutputNeuron())
            self.neurons[i].add_input(self.neurons[hidden_neuron_index])

    def think(self, time_steps, active_input_indexes):
        """activate inputs"""
        for i in active_input_indexes:
            self.neurons[i].change_input(1)

        """think for given number of steps"""
        for time_step in range(time_steps):
            """each neuron should save their inputs. This prohibits computation with newer inputs"""
            for neuron in range(len(self.neurons)):
                self.neurons[neuron].read_inputs()

            """"deactivate input neurons"""
            for i in range(self.num_inputs):
                self.neurons[i].change_input(0)

            """each neuron computes output from just saved inputs"""
            for neuron in range(len(self.neurons)):
                self.neurons[neuron].compute()

            for i in self.output_neuron_index_range:
                if self.neurons[i].output == 1:
                    return i

        """return None if no output was active"""
        return None

    def learn(self, reward):
        for neuron in range(len(self.neurons)):
            self.neurons[neuron].learn(reward)

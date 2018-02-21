from Neuron import Neuron
from InputNeuron import InputNeuron


class Brain:
    """Class for managing Neurons"""

    def __init__(self, num_inputs, num_outputs, learning_rate):
        """initializing neuron objects"""
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.output_callback_function = None
        self.neurons = []
        self.firing_threshold = 0.5

        """initialize input neurons"""
        for i in range(num_inputs):
            self.neurons.append(InputNeuron())

        hidden_neuron_index = num_inputs
        """initialize first hidden neuron with all inputs"""
        self.neurons.append(Neuron(learning_rate))
        for i in range(num_inputs):
            self.neurons[hidden_neuron_index].add_input(self.neurons[i])

        """initialize Output neurons"""
        self.output_neuron_index_range = range(hidden_neuron_index + 1, hidden_neuron_index + 1 + num_outputs)
        for i in self.output_neuron_index_range:
            self.neurons.append(Neuron(learning_rate))
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
            num_neurons_fired = 0
            for neuron in range(len(self.neurons)):
                num_neurons_fired += self.neurons[neuron].compute(self.firing_threshold)

            # adjust new firing_threshold
            self.firing_threshold += 0.02 * (num_neurons_fired / (len(self.neurons) - self.num_inputs) - 0.5)

            for i in self.output_neuron_index_range:
                if self.neurons[i].output == 1:
                    return i - self.output_neuron_index_range[0]

        """return None if no output was active"""
        return None

    def learn(self, reward):
        for neuron in range(len(self.neurons)):
            self.neurons[neuron].learn(reward)

    def terminate_episode(self):
        for neuron in range(len(self.neurons)):
            self.neurons[neuron].reset()


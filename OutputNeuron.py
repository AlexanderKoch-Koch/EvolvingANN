from Neuron import Neuron


class OutputNeuron(Neuron):

    def compute(self):
        """computes neuron outputs with current inputs and stores the result in output"""
        weighted_sum = 0
        for i in range(len(self.inputs) - 1):
            weighted_sum += self.inputs[i] * self.weights[i]

        if weighted_sum > 1:
            self.output = 1
        else:
            self.output = 0

        print(self.output)
        """TODO"""

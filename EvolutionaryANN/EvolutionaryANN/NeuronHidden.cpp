#include "stdafx.h"
#include "NeuronHidden.h"

using namespace std;

NeuronHidden::NeuronHidden(int num_inputs) : Neuron(num_inputs)
{
}

double NeuronHidden::compute(std::vector<double> inputs)
{
	double weighted_sum = sum(inputs);
	double output = sigmoid(weighted_sum);
	last_inputs = inputs;
	last_sum = weighted_sum;
	last_output = weighted_sum;

	for (int i = 0; i < outputs.size(); i++) {
		*outputs[i] = output;
	}
	return output;
}

vector<double> NeuronHidden::learn(double backprop_derivative)
{
	//compute derivative part for backpropagation
	backprop_derivative *= sigmoid_derivative(last_sum);
	
	vector<double> backprop_derivatives;
	//update ech weight
	for (int i = 0; i < weights.size(); i++) {
		backprop_derivatives.push_back(backprop_derivative * weights[i]);
		weights[i] += -learning_rate * backprop_derivative * last_inputs[i];
	}
	return backprop_derivatives;
}

void NeuronHidden::add_output_neuron(NeuronOutput * neuron_output, int index)
{
	connected_output_neurons.push_back(neuron_output);
}

void NeuronHidden::add_output(double * output)
{
	outputs.push_back(output);
}


void NeuronHidden::add_hidden_neuron(NeuronHidden * neuron_hidden, int index)
{
	connected_hidden_neurons.push_back(neuron_hidden);
}

double * NeuronHidden::create_synapse(NeuronHidden * input_neuron)
{
	//add new input
	inputs.push_back(0.0);
	input_neurons.push_back(input_neuron);
	//return address of this new input
	return &inputs[inputs.size() - 1];
}

double * NeuronHidden::create_synapse_input(NeuronInput* input_neuron)
{
	//add new input
	inputs.push_back(0.0);
	input_neurons_input.push_back(input_neuron);
	//return address of this new input
	return &inputs[inputs.size() - 1];
}

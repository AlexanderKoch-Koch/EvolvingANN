#include "stdafx.h"
#include "NeuronInput.h"
#include <vector>

using namespace std;

NeuronInput::NeuronInput(int num_inputs) : Neuron(num_inputs)
{

}

void NeuronInput::setInput(double new_value)
{
	double output = compute(new_value);

	for (int i = 0; i < connected_neurons.size(); i++){
		connected_neurons[i]->setInput(output);
	}
}

double NeuronInput::compute(double input)
{
	return input;
}

void NeuronInput::add_neuron_input(NeuronHidden * neuron_input_variable)
{
	connected_neurons.push_back(neuron_input_variable);
}

void NeuronInput::add_output(double * output)
{
	outputs.push_back(output);
}

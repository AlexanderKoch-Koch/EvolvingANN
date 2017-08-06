#pragma once

#include "stdafx.h"
#include "NeuronOutput.h"
#include "NeuronHidden.h"

using namespace std;

NeuronOutput::NeuronOutput(int num_inputs) : Neuron(num_inputs)
{

}

double NeuronOutput::compute()
{
	double weighted_sum = sum(inputs);
	last_inputs = inputs;
	last_sum = weighted_sum;
	last_output = weighted_sum;
	return weighted_sum;
}

void NeuronOutput::setInput(double new_value, int index)
{
	inputs[index] = new_value;
}

vector<double> NeuronOutput::learn(double output_desired)
{
	//compute derivative part for backpropagation
	double derivative = (last_output - output_desired) * sigmoid_derivative(last_sum);

	vector<double> backprop_derivatives;
	//update ech weight
	for (unsigned int i = 0; i < weights.size(); i++) {
		backprop_derivatives.push_back(derivative * weights[i]);
		weights[i] += -learning_rate * derivative * last_inputs[i];
	}

	
	return backprop_derivatives;
}

double * NeuronOutput::add_input()
{
	//add new input
	inputs.push_back(0.0);
	//return address of this new input
	return &inputs[inputs.size() - 1];
}

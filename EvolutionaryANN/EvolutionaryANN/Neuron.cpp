#include "stdafx.h"
#include "Neuron.h"
#include <cmath>
#include <cstdlib>



Neuron::Neuron(int num_inputs)
{
	learning_rate = 0.01f;
	for(int i = 0; i < num_inputs; i++){
		weights.push_back((rand() % 100 + 1) / 100.0);
	}
}

double Neuron::sigmoid(double variable) {
	const double euler = 2.71828;
	return 1 / (1 + pow(euler, (-1 * variable)));
}

double Neuron::sigmoid_derivative(double variable)
{
	return sigmoid(variable) * (1 - sigmoid(variable));
}

double Neuron::sum(std::vector<double> inputs)
{
	double weighted_sum = 0;
	for (int i = 0; i < inputs.size(); i++) {
		weighted_sum += inputs[i] * weights[i];
	}

	return weighted_sum;
}


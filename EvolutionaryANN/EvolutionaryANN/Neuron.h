#pragma once

#include <vector>



class Neuron
{
public:
	Neuron(int num_inputs);
	double sum(std::vector<double> inputs);
	double sigmoid(double variable);
	double sigmoid_derivative(double variable);

protected:
	std::vector<double> weights;
	double learning_rate;
	std::vector<double> last_inputs;
	double last_output;
	double last_sum;
};
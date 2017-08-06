#pragma once

#include "Neuron.h"
#include "NeuronHidden.h"

class NeuronOutput : public Neuron{

public:
	NeuronOutput(int num_inputs);
	double compute();
	void setInput(double new_value, int index);
	std::vector<double> learn(double output_desired);
	double * add_input();

private:
	std::vector<double> inputs;
	std::vector<double *> output_connected_inputs;
};
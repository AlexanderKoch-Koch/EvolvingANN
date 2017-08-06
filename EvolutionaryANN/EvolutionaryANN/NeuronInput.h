#pragma once

#include "Neuron.h"
#include "NeuronHidden.h"
#include <vector>



class NeuronInput : public Neuron {

public:
	NeuronInput(int num_inputs);
	void setInput(double new_value);
	double compute(double input);
	void add_output(double* output);

private:
	std::vector<double*> output_connected_inputs;

};
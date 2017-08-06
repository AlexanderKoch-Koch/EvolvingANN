#pragma once

#include "Neuron.h"
#include "NeuronHidden.h"
#include <vector>



class NeuronInput : public Neuron {

public:
	NeuronInput(int num_inputs);
	void setInput(double new_value);
	double compute(double input);
	void add_neuron_input(NeuronHidden * neuron_input_variable);
	void add_output(double* output);

private:
	std::vector<NeuronHidden *> connected_neurons;
	std::vector<double*> outputs;

};
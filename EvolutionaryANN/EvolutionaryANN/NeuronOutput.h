#pragma once

#include "Neuron.h"
#include "NeuronHidden.h"

class NeuronOutput : public Neuron{

public:
	NeuronOutput(int num_inputs);
	double compute();
	void setInput(double new_value, int index);
	std::vector<double> learn(double output_desired);
	double * create_synapse(NeuronHidden * input_neuron);

private:
	std::vector<double> inputs;
	std::vector<NeuronHidden *> input_neurons;
};
#pragma once

#include "Neuron.h"
#include "NeuronOutput.h"
#include "NeuronInput.h"
#include <vector>



class NeuronHidden : public Neuron {

public:
	NeuronHidden(int num_inputs);
	/*forawrd pass*/
	double compute(std::vector<double> inputs);
	/*updates weights and returns factor for backpropagation*/
	std::vector<double> learn(double backprop_derivative);

	void add_hidden_neuron(NeuronHidden *neuron_hidden, int index);
	void add_output_neuron(NeuronOutput *neuron_output, int index);
	void add_output(double * output);
	double * create_synapse(NeuronHidden * input_neuron);
	double * create_synapse_input(NeuronInput* input_neuron);

private:
	std::vector<NeuronHidden*> connected_hidden_neurons;
	std::vector<NeuronOutput*> connected_output_neurons;
	std::vector<double*> outputs;

	std::vector<double> inputs;
	std::vector<NeuronHidden *> input_neurons;
	std::vector<NeuronInput *> input_neurons_input;
};
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

	void add_output(double * output);
	double * add_input();

private:
	std::vector<double> inputs;
	std::vector<double *> output_connected_inputs;
};
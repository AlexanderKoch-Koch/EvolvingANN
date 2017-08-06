#pragma once

#include <vector>
#include "NeuronInput.h"
#include "NeuronOutput.h"
#include "NeuronHidden.h"

class NeuralNet
{
public:

	NeuralNet(int inputs, int outputs, int neurons_number);
	double train(std::vector<double> inputs, std::vector<double> outputs_desired);
	std::vector<double> compute(std::vector<double> inputs);

private:
	std::vector<NeuronInput> neurons_input;
	std::vector<NeuronHidden> neurons_hidden;
	std::vector<NeuronOutput> neurons_output;
	

};
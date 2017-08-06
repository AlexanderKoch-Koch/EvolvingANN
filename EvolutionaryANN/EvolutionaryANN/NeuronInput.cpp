#include "stdafx.h"
#include "NeuronInput.h"
#include <vector>

using namespace std;

NeuronInput::NeuronInput(int num_inputs) : Neuron(num_inputs)
{

}


void NeuronInput::compute()
{
	for (int i = 0; i < output_connected_inputs.size(); i++) {
		*output_connected_inputs[i] = input;
	}
}


void NeuronInput::add_output(double * output)
{
	output_connected_inputs.push_back(output);
}

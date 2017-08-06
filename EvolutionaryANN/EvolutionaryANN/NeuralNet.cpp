#include "stdafx.h"
#include "NeuralNet.h"
#include <vector>
#include "NeuronInput.h"
#include "NeuronOutput.h"
#include "NeuronHidden.h"
#include <cmath>

using namespace std;

NeuralNet::NeuralNet(int inputs, int outputs, int neurons_number)
{	
	for (int i = 0; i < outputs; i++)
	{
		neurons_output.push_back(NeuronOutput(2));
	}

	neurons_hidden.push_back(NeuronHidden(1));
	for (int i = 0; i < outputs; i++)
	{
		neurons_hidden[0].add_output(neurons_output[i].create_synapse(&neurons_hidden[0]));
		
	}
	for (int i = 0; i < inputs; i++)
	{
		neurons_input.push_back(NeuronInput(1));
		neurons_input[i].add_output(neurons_hidden[0].create_synapse_input(&neurons_input[i]));
	}
}

double NeuralNet::train(vector<double> inputs, vector<double> outputs_desired)
{
	vector<double> outputs = compute(inputs);
	vector<double> l2_backprops = neurons_output[0].learn(outputs_desired[0]);
	vector<double> l1_backprops = neurons_hidden[0].learn(l2_backprops[0]);

	//compute error
	double error = 0.0;
	for (int i = 0; i < outputs.size(); i++) {
		error += pow(outputs[i] - outputs_desired[i], 2);
	}
	return error;
}

vector<double> NeuralNet::compute(vector<double> inputs)
{

	/*double l1_0 = neurons_input[0].compute(inputs[0]);

	for (int i = 0; i < neurons_hidden.size(); i++) {
		neurons_hidden[i].compute();
	}

	*/vector<double> outputs;/*
	for (int i = 0; i < neurons_output.size(); i++) {
		outputs.push_back(neurons_output[i].compute());
	}*/

	return outputs;
}

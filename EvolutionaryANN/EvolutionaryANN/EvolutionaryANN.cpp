// EvolutionaryANN.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include "NeuralNet.h"

using namespace std;

int main()
{
	//test
	NeuralNet ann = NeuralNet(1, 1, 2);
	vector<double> inputs;
	inputs.push_back(1);
	cout << ann.compute(inputs)[0];

	//training
	/*vector<double> outputs_desired;
	outputs_desired.push_back(5.2);
	for (int i = 0; i < 1000; i++) {
		cout << ann.train(inputs, outputs_desired) << endl;
	}*/

	cin.get();	//keep window open
	return 0;
}


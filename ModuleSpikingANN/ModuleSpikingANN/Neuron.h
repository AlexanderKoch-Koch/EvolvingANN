#ifndef NEURON_H
#define NEURON_H
#include "Parameters.h"
#include "Synapse.h"
/**
Compute if neuron fires
@param inputs current inputs to the Neuron
@param weights current weights of the num_inputs
@param synapse_activities increase if pre and postsynaptic neurons fire
@return 1 if fires, otherwise 0
*/
void compute(struct Synapse **neurons, int num_neurons, int num_synapses_per_neuron, int *neuron_outputs);

void read(struct Synapse **neurons, int num_neurons, int num_synapses_per_neuron);

void learn(struct Synapse **neurons, int num_neurons, int num_synapses_per_neuron, float reward);

#endif

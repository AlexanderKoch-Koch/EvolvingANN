#ifndef NEURON_H
#define NEURON_H
#include "Parameters.h"
#include "Synapse.h"

/**
Compute if neuron fires
@param syanpses contains all synapse values in the brain
@param neuron_outputs int array for storing neuron outputs
@param p hyperparamters of the brain
@return 1 if fires, otherwise 0
*/
void compute(struct Synapse **synapses, int *neuron_outputs, struct Parameters *p);

void read(struct Synapse **synapses, struct Parameters *p);

void learn(struct Synapse **synapses, float reward, struct Parameters *p);

#endif

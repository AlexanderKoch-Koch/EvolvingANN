#ifndef NEURON_H
#define NEURON_H
#include "Parameters.h"
/**
Compute if neuron fires
@param inputs current inputs to the Neuron
@param weights current weights of the num_inputs
@param synapse_activities increase if pre and postsynaptic neurons fire
@return 1 if fires, otherwise 0
*/
void compute(
  int inputs[5][NUM_SYNAPSES_PER_NEURON],
  float weights[5][NUM_SYNAPSES_PER_NEURON],
  int neuron_outputs[5]
);

void read(
  int inputs[5][NUM_SYNAPSES_PER_NEURON],
  int *p_presynaptic_neuron_outputs[5][NUM_SYNAPSES_PER_NEURON]
);

void learn(
  float weights[5][NUM_SYNAPSES_PER_NEURON],
  float synapse_activities[5][NUM_SYNAPSES_PER_NEURON],
  float reward
);

void tag_synapse(
  float synapse_activities[5][NUM_SYNAPSES_PER_NEURON],
  int neuron_outputs[5],
  int neuron_inputs[5][NUM_SYNAPSES_PER_NEURON]
);

#endif

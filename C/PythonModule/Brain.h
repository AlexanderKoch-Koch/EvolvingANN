#ifndef BRAIN_H
#define BRAIN_H

/**
initialize varibles for num_neurons
@param num_neurons number odf neurons which should be created
@param num_inputs number of binary inputs to the brain
@param num_outputs number of binary ouytptuts from te brain
*/
void init_brain(struct Parameters parameters);

/**
do one think step- read inputs, compute, return outputs
@param inputs new binary inputs for brain
@param len_inputs length of @param inputs array
@param num_outputs this variable will be set to the number of outputs returned
*/
int * think(float *inputs, int len_inputs, int *num_outputs);

/**
adjust the neuron weights
@param reward current reward for brain
*/
void process_reward(float reward);

/**
free memory used by neuron arrays
*/
void release_memory();

/**
reset synapse_activities. Weights won't be changed.
*/
void reset_memory();

#endif

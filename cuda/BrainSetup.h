#ifndef BRAINSETUP_H
#define BRAINSETUP_H

__global__ void init_random_seed(unsigned int seed, curandState_t d_curand_state);

__global__ void init_synapses(struct Synapse *d_synapses, size_t pitch, int *d_neuron_outputs, int *d_brain_inputs, curandState_t d_curand_state);

__global__ void init_neurons(int *d_neuron_outputs, float *d_weighted_sums);

#endif
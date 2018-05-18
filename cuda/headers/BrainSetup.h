#ifndef BRAINSETUP_H
#define BRAINSETUP_H

#include <cuda.h>
#include <curand_kernel.h>


__global__ void init_random_seed(unsigned int seed, curandState_t *d_curand_state);

__global__ void init_synapses(struct Synapse *d_synapses, size_t pitch, int *d_neuron_outputs, int *d_brain_inputs, curandState_t *d_curand_state);

__global__ void init_t1_synapses(struct Synapse *d_synapses, size_t t1_pitch, size_t pitch, int *d_t1_neuron_outputs, int *d_neuron_outputs, int *d_brain_inputs, curandState_t *d_curand_state);

#endif
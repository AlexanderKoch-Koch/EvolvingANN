#ifndef COMPUTE_H
#define COMPUTE_H

#include <cuda.h>
#include <curand_kernel.h>

__global__ void compute(struct Synapse *d_synapses, int *d_neuron_outputs, size_t pitch, curandState_t *d_curand_states, struct Parameters *d_parameters);

__global__ void compute_t1(struct Synapse *d_t1_synapses, int *d_t1_neuron_outputs, size_t t1_pitch, curandState_t *d_t1_curand_states, struct Parameters *d_parameters);

__global__ void read(struct Synapse *d_synapses, size_t pitch);

__global__ void read_t1(struct Synapse *d_t1_synapses, size_t t1_pitch);

__global__ void learn(struct Synapse *d_synapses, float learning_factor, size_t pitch, int *d_neuron_outputs, int *d_brain_inputs, curandState_t *d_curand_state);

__global__ void learn_t1(struct Synapse *d_t1_synapses, float reward, size_t t1_pitch, int *d_t1_neuron_outputs, int *d_neuron_outputs, int *d_brain_inputs, curandState_t *d_t1_curand_states);

__global__ void reset_synapses(struct Synapse *d_synapses, float *d_weighted_sums, size_t pitch);

__global__ void update_parameters(struct Parameters *d_parameters);

#endif
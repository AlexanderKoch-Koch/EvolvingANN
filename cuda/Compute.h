#ifndef COMPUTE_H
#define COMPUTE_H

__global__ void compute_synapses(struct Synapse *d_synapses, float *d_weighted_sums, size_t pitch);

__global__ void compute_neurons(int *d_neuron_outputs, float *d_weighted_sums);

__global__ void learn(struct Synapse *d_synapses, float reward, size_t pitch);

#endif
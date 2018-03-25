#ifndef COMPUTE_H
#define COMPUTE_H

__global__ void compute_synapses(struct Synapse *d_synapses, float *d_weighted_sums, size_t pitch);

__global__ void compute_neurons(int *d_neuron_outputs, float *d_weighted_sums);

#endif
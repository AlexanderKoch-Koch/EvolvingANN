#ifndef VISUALIZE_H
#define VISUALIZE_H

__global__ void printSynapses(struct Synapse *d_synapses, size_t pitch);

__global__ void printNeurons(int *d_neuron_outputs, float *d_weighted_sums);

__global__ void print_parameters(struct Parameters *d_parameters);

__global__ void print_synapse_stats(struct Synapse *d_synapses, size_t pitch);

void neuron_stats(int *d_neuron_outputs, unsigned long step);

void synapse_stats(struct Synapse *d_synapses, size_t pitch, unsigned long step);

#endif
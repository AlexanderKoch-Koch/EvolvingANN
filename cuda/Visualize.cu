#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "Synapse.h"
#include "Hyperparameters.h"



__global__ void printSynapses(struct Synapse *d_synapses, size_t pitch){
    int synapse = blockIdx.x*blockDim.x + threadIdx.x;
    int neuron = blockIdx.y*blockDim.y + threadIdx.y;

    struct Synapse *row_a = (struct Synapse *) ((char*)d_synapses + neuron * pitch);
    printf("neuron: %d, synapse: %d, weight: %.2f, activity: %.2f, input: %d\n", neuron, synapse, row_a[synapse].weight, row_a[synapse].activity, row_a[synapse].input);
}

__global__ void printNeurons(int *d_neuron_outputs, float *d_weighted_sums){
    int neuron = blockIdx.x*blockDim.x + threadIdx.x;
    printf("neuron: %d, weighted sum: %.2f, output: %d\n", neuron, d_weighted_sums[neuron], d_neuron_outputs[neuron]);
}


__global__ void computeNeurons(int *d_neuron_outputs, float *d_weighted_sums){
    int neuron = blockIdx.x*blockDim.x + threadIdx.x;
    
    
}
#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "Synapse.h"
#include "Hyperparameters.h"


__global__ void compute_synapses(struct Synapse *d_synapses, float *d_weighted_sums, size_t pitch){
    int synapse = blockIdx.x*blockDim.x + threadIdx.x;
    int neuron = blockIdx.y*blockDim.y + threadIdx.y;

    struct Synapse *neuron_array = (struct Synapse *) ((char*)d_synapses + neuron * pitch);
    float sum = neuron_array[synapse].weight *  (*neuron_array[synapse].p_presynaptic_output);
    atomicAdd(&d_weighted_sums[neuron], sum);
    //printf("neuron: %d, synapse: %d,  adding %d\n", neuron, synapse, *neuron_array[synapse].p_presynaptic_output);
}

__global__ void compute_neurons(int *d_neuron_outputs, float *d_weighted_sums){
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(d_weighted_sums[neuron] >= THRESHOLD){
        d_neuron_outputs[neuron] = 1;
        printf("firing :)");
    }else{
        d_neuron_outputs[neuron] = 0;
    }
    
    //reset weighted sum
    d_weighted_sums[neuron] = 0.0;
}
#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "Synapse.h"
#include "BrainSetup.h"
#include "Visualize.h"
#include "Compute.h"
#include "Hyperparameters.h"

dim3 synapses_dim(NUM_SYNAPSES_PER_NEURON, NUM_NEURONS, 1);
size_t dev_pitch;
struct Synapse *d_synapses;
int *d_neuron_outputs;
float *d_weighted_sums;
int *d_brain_inputs;
curandState_t *d_curand_state;

void init(){
    cudaMalloc(&d_curand_state, sizeof(curandState_t));
    init_random_seed<<<1, 1>>>(1, d_curand_state);
    //allocate memory on the device
    cudaMalloc(&d_brain_inputs, sizeof(int) * NUM_INPUTS);
    cudaMalloc(&d_weighted_sums, sizeof(float) * NUM_NEURONS);
    cudaMalloc(&d_neuron_outputs, sizeof(int) * NUM_NEURONS);
    cudaMallocPitch(&d_synapses, &dev_pitch, NUM_SYNAPSES_PER_NEURON * sizeof(struct Synapse), NUM_NEURONS);
    
    // initialize brain
    init_synapses<<<1, synapses_dim>>>(d_synapses, dev_pitch, d_neuron_outputs, d_brain_inputs, d_curand_state);
    init_neurons<<<1, NUM_NEURONS>>>(d_neuron_outputs, d_weighted_sums);
    cudaDeviceSynchronize();
    
    printSynapses<<<1, synapses_dim>>>(d_synapses, dev_pitch);
    printNeurons<<<1, NUM_NEURONS>>>(d_neuron_outputs, d_weighted_sums);
    cudaDeviceSynchronize();
}


int* think(int *inputs){
    //set brain inputs
    cudaMemcpy(d_brain_inputs, inputs, sizeof(int) * NUM_INPUTS, cudaMemcpyHostToDevice);
    
    //sum up the inputs
    compute_synapses<<<1, synapses_dim>>>(d_synapses, d_weighted_sums, dev_pitch);
    cudaDeviceSynchronize();
    
    //decide Threshold
    compute_neurons<<<1, NUM_NEURONS>>>(d_neuron_outputs, d_weighted_sums);
    cudaDeviceSynchronize();
    tag_synapses<<<1, synapses_dim>>>(d_synapses, d_neuron_outputs, dev_pitch);
    //copy results back to host
    int *outputs = (int*) malloc(sizeof(int) * NUM_OUTPUTS);
    cudaMemcpy(outputs, d_neuron_outputs, sizeof(int) * NUM_OUTPUTS, cudaMemcpyDeviceToHost);
    return outputs;
}

void process_reward(float reward){
    printf("reward is %.2f", reward);
}

void reset_memory(){
    reset_synapses<<<1, synapses_dim>>>(d_synapses, d_weighted_sums, dev_pitch);
}

void release_memory(){
    cudaFree(d_brain_inputs);
    cudaFree(d_neuron_outputs);
    cudaFree(d_synapses);
    cudaFree(d_weighted_sums);
}
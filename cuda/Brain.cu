#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "Synapse.h"
#include "BrainSetup.h"
#include "Visualize.h"
#include "Compute.h"
#include "Hyperparameters.h"




void init(){
    //initialize curand
    printf("hello world1");
    curandState_t d_curand_state;
    init_random_seed<<<1, 1>>>(time(0), d_curand_state);
    
    //allocate memory for neurons
    size_t dev_pitch;
    struct Synapse *d_synapses;
    int *d_neuron_outputs;
    float *d_weighted_sums;
    int *d_brain_inputs;
    cudaMalloc(&d_brain_inputs, sizeof(int) * NUM_INPUTS);
    cudaMalloc(&d_weighted_sums, sizeof(float) * NUM_NEURONS);
    cudaMalloc(&d_neuron_outputs, sizeof(int) * NUM_NEURONS);
    cudaMallocPitch(&d_synapses, &dev_pitch, NUM_SYNAPSES_PER_NEURON * sizeof(struct Synapse), NUM_NEURONS);
    
    dim3 synapses_dim(NUM_SYNAPSES_PER_NEURON, NUM_NEURONS, 1);

    // initialize brain
    init_synapses<<<1, synapses_dim>>>(d_synapses, dev_pitch, d_neuron_outputs, d_brain_inputs, d_curand_state);
    init_neurons<<<1, NUM_NEURONS>>>(d_neuron_outputs, d_weighted_sums);
    cudaDeviceSynchronize();
    
    //set brain inputs
    int inputs[] = {1, 0, 1, 1};
    cudaMemcpy(d_brain_inputs, inputs, sizeof(int) * NUM_INPUTS, cudaMemcpyHostToDevice);
    
    compute_synapses<<<1, synapses_dim>>>(d_synapses, d_weighted_sums, dev_pitch);
    cudaDeviceSynchronize();
    
    compute_neurons<<<1, NUM_NEURONS>>>(d_neuron_outputs, d_weighted_sums);
    cudaDeviceSynchronize();
    
    printSynapses<<<1, synapses_dim>>>(d_synapses, dev_pitch);
    printNeurons<<<1, NUM_NEURONS>>>(d_neuron_outputs, d_weighted_sums);
    cudaDeviceSynchronize();
}
#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "Synapse.h"
#include "BrainSetup.h"
#include "Visualize.h"
#include "Compute.h"
#include "Hyperparameters.h"
#include "Parameters.h"



struct Parameters *d_parameters;
int *d_brain_inputs;
unsigned long iteration_counter = 0;
// main ANN
struct Synapse *d_synapses;
size_t synapses_pitch;
int *d_neuron_outputs;
float *d_weighted_sums;
dim3 block_dim(512, 1, 1);
dim3 grid_dim((NUM_NEURONS + block_dim.x - 1) / block_dim.x);
curandState_t *d_curand_states;

//trainer ANN
struct Synapse *d_t1_synapses;
size_t t1_synapses_pitch;
int *d_t1_neuron_outputs;
float *d_t1_weighted_sums;
dim3 t1_block_dim(512, 1, 1);
dim3 t1_grid_dim((NUM_T1_NEURONS + t1_block_dim.x - 1) / t1_block_dim.x);
curandState_t *d_t1_curand_states;


void init(){
    //mark output start
    printf("#################################################################################################");
    printf("synapses memory usage: %zu Bytes", sizeof(struct Synapse) * NUM_NEURONS * NUM_SYNAPSES_PER_NEURON);
    printf("num_neurons: %d block size: %d grid size: %d", NUM_NEURONS, block_dim.x, grid_dim.x);
    
    //allocate memory on the device
    cudaMalloc(&d_curand_states, sizeof(curandState_t) * NUM_NEURONS);
    cudaMalloc(&d_t1_curand_states, sizeof(curandState_t) * NUM_T1_NEURONS);
    cudaMalloc(&d_parameters, sizeof(struct Parameters));
    cudaMalloc(&d_brain_inputs, sizeof(int) * NUM_INPUTS);
    cudaMalloc(&d_weighted_sums, sizeof(float) * NUM_NEURONS);
    cudaMalloc(&d_t1_weighted_sums, sizeof(float) * NUM_T1_NEURONS);
    cudaMalloc(&d_neuron_outputs, sizeof(int) * NUM_NEURONS);
    cudaMalloc(&d_t1_neuron_outputs, sizeof(int) * NUM_T1_NEURONS);
    cudaMallocPitch(&d_synapses, &synapses_pitch, NUM_SYNAPSES_PER_NEURON * sizeof(struct Synapse), NUM_NEURONS);
    cudaMallocPitch(&d_t1_synapses, &t1_synapses_pitch, NUM_SYNAPSES_PER_NEURON * sizeof(struct Synapse), NUM_T1_NEURONS);
    
    //init curand states
    init_random_seed<<<grid_dim, block_dim>>>(time(NULL), d_curand_states);
    init_random_seed<<<t1_grid_dim, t1_block_dim>>>(time(NULL), d_t1_curand_states);
    
    //copy parameter struct to device
    struct Parameters start_parameters;
    start_parameters.threshold_randomness_factor = THRESHOLD_RANDOMNESS_FACTOR_START;
    cudaMemcpy(d_parameters, &start_parameters, sizeof(struct Parameters), cudaMemcpyHostToDevice);
    
    // initialize brain
    init_synapses<<<grid_dim, block_dim>>>(d_synapses, synapses_pitch, d_neuron_outputs, d_brain_inputs, d_curand_states);
    init_t1_synapses<<<t1_grid_dim, t1_block_dim>>>(d_t1_synapses, t1_synapses_pitch, synapses_pitch, d_t1_neuron_outputs, d_neuron_outputs, d_brain_inputs, d_t1_curand_states);
}


int* think(int *inputs){
    //set brain inputs
    cudaMemcpy(d_brain_inputs, inputs, sizeof(int) * NUM_INPUTS, cudaMemcpyHostToDevice);

    //read
    update_parameters<<<1, 1>>>(d_parameters);
    read<<<grid_dim, block_dim>>>(d_synapses, synapses_pitch);
    cudaDeviceSynchronize();
    
    //compute
    compute<<<grid_dim, block_dim>>>(d_synapses, d_neuron_outputs, synapses_pitch, d_curand_states, d_parameters);
    cudaDeviceSynchronize();

    if(iteration_counter % 5000 == 0){
        //show info
        printf("iteration: %ld\n", iteration_counter);
        neuron_stats(d_neuron_outputs);
        print_synapse_stats<<<grid_dim, block_dim>>>(d_synapses, synapses_pitch);
        printSynapses<<<grid_dim, block_dim>>>(d_synapses, synapses_pitch);
        print_parameters<<<1, 1>>>(d_parameters);
    }
    
    //get brain outputs
    int *outputs = (int*) malloc(sizeof(int) * NUM_OUTPUTS);
    cudaMemcpy(outputs, d_neuron_outputs, sizeof(int) * NUM_OUTPUTS, cudaMemcpyDeviceToHost);
    iteration_counter++;
    return outputs;
}

void process_reward(float reward){
    learn<<<grid_dim, block_dim>>>(d_synapses, reward, synapses_pitch, d_neuron_outputs, d_brain_inputs, d_curand_states);
}

void reset_memory(){
    reset_synapses<<<grid_dim, block_dim>>>(d_synapses, d_weighted_sums, synapses_pitch);
}

void release_memory(){
    cudaFree(d_brain_inputs);
    cudaFree(d_neuron_outputs);
    cudaFree(d_synapses);
    cudaFree(d_weighted_sums);
}
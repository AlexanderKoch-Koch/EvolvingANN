#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "Synapse.h"
#include "Hyperparameters.h"


__global__ void init_random_seed(unsigned int seed, curandState_t *d_curand_state) {

  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              blockIdx.x, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              d_curand_state);
}


__global__ void init_synapses(struct Synapse *d_synapses, size_t pitch, int *d_neuron_outputs, int *d_brain_inputs, curandState_t *d_curand_state){
    int synapse = blockIdx.x*blockDim.x + threadIdx.x;
    int neuron = blockIdx.y*blockDim.y + threadIdx.y;

    skipahead(synapse * (neuron + 1), d_curand_state);
    float new_weight = curand_uniform(d_curand_state);
    
    struct Synapse *neuron_array = (struct Synapse *) ((char*)d_synapses + neuron * pitch);
    neuron_array[synapse].weight = new_weight;
    
    int rand_input = curand(d_curand_state) % (NUM_NEURONS + NUM_INPUTS);
    if(rand_input < NUM_NEURONS){
        //connect to other neuron
        neuron_array[synapse].p_presynaptic_output = &d_neuron_outputs[rand_input];
    }else{
        //connect to brain input
        neuron_array[synapse].p_presynaptic_output = &d_brain_inputs[rand_input - NUM_NEURONS];
    }
    //printf("neuron: %d, synapse: %d, new_weight: %.2f\n", neuron, synapse,  new_weight);
}


__global__ void init_neurons(int *d_neuron_outputs, float *d_weighted_sums){
    int neuron = blockIdx.x*blockDim.x + threadIdx.x;
    
    d_neuron_outputs[neuron] = 0;
    d_weighted_sums[neuron] = 0;
}
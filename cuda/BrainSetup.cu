#include "BrainSetup.h"
#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "Synapse.h"
#include "Hyperparameters.h"


__global__ void init_random_seed(unsigned int seed, curandState_t *d_curand_state) {

    /* we have to initialize the state */
    curand_init(seed, blockIdx.x, 0, d_curand_state);
}


__global__ void init_synapses(struct Synapse *d_synapses, size_t pitch, int *d_neuron_outputs, int *d_brain_inputs, curandState_t *d_curand_state){
    int neuron = blockIdx.x*blockDim.x + threadIdx.x;
    if(neuron < NUM_NEURONS){

        d_neuron_outputs[neuron] = curand(d_curand_state) % 2;
    
        struct Synapse *neuron_array = (struct Synapse *) ((char*)d_synapses + neuron * pitch);
        skipahead(neuron, d_curand_state);
        
        for(int synapse = 0; synapse < NUM_SYNAPSES_PER_NEURON; synapse++){
            create_synapse(&neuron_array[synapse], d_neuron_outputs, d_brain_inputs, d_curand_state);
        }
    }
}

__device__ void create_synapse(struct Synapse *d_synapse, int *d_neuron_outputs, int *d_brain_inputs, curandState_t *d_curand_state){
    float new_weight = curand_uniform(d_curand_state) + MIN_START_WEIGHT;
            
    d_synapse->weight = new_weight;
            
    int rand_input = curand(d_curand_state) % (NUM_NEURONS + NUM_INPUTS);
    if(rand_input < NUM_NEURONS){
        //connect to other neuron
        d_synapse->p_presynaptic_output = &d_neuron_outputs[rand_input];
    }else{
        //connect to brain input
        d_synapse->p_presynaptic_output = &d_brain_inputs[rand_input - NUM_NEURONS];
    }
}

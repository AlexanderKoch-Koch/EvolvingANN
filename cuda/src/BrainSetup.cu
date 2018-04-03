#include "BrainSetup.h"
#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "Synapse.h"
#include "Hyperparameters.h"


__global__ void init_random_seed(unsigned int seed, curandState_t *d_curand_state) {
    curand_init(seed, blockIdx.x, 0, d_curand_state);
}


__global__ void init_synapses(struct Synapse *d_synapses, size_t pitch, int *d_neuron_outputs, int *d_brain_inputs, curandState_t *d_curand_state){
    int neuron = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(neuron < NUM_NEURONS){
        skipahead(neuron, d_curand_state);
        //set randomly 3/13 of the outputs to 1
        d_neuron_outputs[neuron] = curand(d_curand_state) % 13 / 10;

        struct Synapse *neuron_array = (struct Synapse *) ((char*)d_synapses + neuron * pitch);
        
        for(int synapse = 0; synapse < NUM_SYNAPSES_PER_NEURON; synapse++){
            //random weight between MIN_START_WEIGHT and MIN_START_WEIGHT + 1
            float new_weight = curand_uniform(d_curand_state) + MIN_START_WEIGHT;
            //printf("neuron %d synapse: %d weight %.2f\n", neuron, synapse, new_weight);
            neuron_array[synapse].weight = new_weight;
            //random input
            int rand_input = curand(d_curand_state) % (NUM_NEURONS + NUM_INPUTS);
            if(rand_input < NUM_NEURONS){
                //connect to other neuron
                neuron_array[synapse].p_presynaptic_output = &d_neuron_outputs[rand_input];
            }else{
                //connect to brain input
                neuron_array[synapse].p_presynaptic_output = &d_brain_inputs[rand_input - NUM_NEURONS];
            }
        }
    }
}

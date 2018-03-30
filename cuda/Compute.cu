#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "Synapse.h"
#include "Hyperparameters.h"



__global__ void compute_neurons(struct Synapse *d_synapses, int *d_neuron_outputs, size_t pitch){
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if(neuron < NUM_NEURONS){
        struct Synapse *neuron_array = (struct Synapse *) ((char*)d_synapses + neuron * pitch);
        float weighted_sum = 0.0;
        for(int synapse = 0; synapse < NUM_SYNAPSES_PER_NEURON; synapse++){
            weighted_sum += neuron_array[synapse].input * neuron_array[synapse].weight;
        }
        
        if(weighted_sum >= THRESHOLD){
            d_neuron_outputs[neuron] = 1;
        }else{
            d_neuron_outputs[neuron] = 0;
        }
        for(int synapse = 0; synapse < NUM_SYNAPSES_PER_NEURON; synapse++){
            neuron_array[synapse].activity *= ACTIVITY_DISCOUNT_FACTOR;
            neuron_array[synapse].activity += neuron_array[synapse].input * d_neuron_outputs[neuron] * neuron_array[synapse].weight;
        }
    }
}


__global__ void read(struct Synapse *d_synapses, size_t pitch){
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if(neuron < NUM_NEURONS){

        struct Synapse *neuron_array = (struct Synapse *) ((char*)d_synapses + neuron * pitch);
        
        for(int synapse = 0; synapse < NUM_SYNAPSES_PER_NEURON; synapse++){
            neuron_array[synapse].input = (*neuron_array[synapse].p_presynaptic_output);
        }
    }
}


__global__ void learn(struct Synapse *d_synapses, float reward, size_t pitch, int *d_neuron_outputs, int *d_brain_inputs, curandState_t *d_curand_state){
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if(neuron < NUM_NEURONS){

        struct Synapse *neuron_array = (struct Synapse *) ((char*)d_synapses + neuron * pitch);
        for(int synapse = 0; synapse < NUM_SYNAPSES_PER_NEURON; synapse++){
            neuron_array[synapse].weight += LEARNING_RATE * reward * neuron_array[synapse].activity;
            //randomly reconnect if weight too small
            if(fabsf(neuron_array[synapse].weight) < MIN_ABS_WEIGHT){
                printf("reconnect");
                float new_weight = curand_uniform(d_curand_state) + MIN_START_WEIGHT;
            
                neuron_array[synapse].weight = new_weight;
            
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
}


__global__ void reset_synapses(struct Synapse *d_synapses, float *d_weighted_sums, size_t pitch){
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;

    if(neuron < NUM_NEURONS){
        struct Synapse *neuron_array = (struct Synapse *) ((char*)d_synapses + neuron * pitch);
        
        for(int synapse = 0; synapse < NUM_SYNAPSES_PER_NEURON; synapse++){
            neuron_array[synapse].input = 0;
            neuron_array[synapse].activity = 0;
        }
    }
}
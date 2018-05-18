#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "Synapse.h"
#include "Hyperparameters.h"
#include "Parameters.h"
#include <math.h>



__global__ void compute(struct Synapse *d_synapses, int *d_neuron_outputs, size_t pitch, curandState_t *d_curand_states, struct Parameters *d_parameters){
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if(neuron < NUM_NEURONS){
        struct Synapse *neuron_array = (struct Synapse *) ((char*)d_synapses + neuron * pitch);
        float weighted_sum = 0.0;
        for(int synapse = 0; synapse < NUM_SYNAPSES_PER_NEURON; synapse++){
            weighted_sum += neuron_array[synapse].input * neuron_array[synapse].weight;
        }
        float random = d_parameters->threshold_randomness_factor * curand_normal(&d_curand_states[neuron]);
        //printf("weighted_sum: %.2f, random_add: %.2f  ", weighted_sum, random);
        if(weighted_sum + random >= THRESHOLD){
            d_neuron_outputs[neuron] = 1;
            //printf("firing");
        }else{
            d_neuron_outputs[neuron] = 0;
        }
        for(int synapse = 0; synapse < NUM_SYNAPSES_PER_NEURON; synapse++){
            neuron_array[synapse].activity *= ACTIVITY_DISCOUNT_FACTOR;
            //neuron_array[synapse].activity += neuron_array[synapse].input * d_neuron_outputs[neuron] * neuron_array[synapse].weight;
            //neuron_array[synapse].activity += neuron_array[synapse].input * fabs(neuron_array[synapse].weight) * (d_neuron_outputs[neuron] - 0.5);
            neuron_array[synapse].activity += neuron_array[synapse].input * (d_neuron_outputs[neuron] - 0.5);
        }
    }
}


__global__ void compute_t1(struct Synapse *d_t1_synapses, int *d_t1_neuron_outputs, size_t t1_pitch, curandState_t *d_t1_curand_states, struct Parameters *d_parameters){
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if(neuron < NUM_T1_NEURONS){
        struct Synapse *neuron_array = (struct Synapse *) ((char*)d_t1_synapses + neuron * t1_pitch);
        float weighted_sum = 0.0;
        for(int synapse = 0; synapse < NUM_SYNAPSES_PER_NEURON; synapse++){
            weighted_sum += neuron_array[synapse].input * neuron_array[synapse].weight;
        }
        float random = d_parameters->threshold_randomness_factor * curand_normal(&d_t1_curand_states[neuron]);
        //printf("weighted_sum: %.2f, random_add: %.2f  ", weighted_sum, random);
        if(weighted_sum + random >= THRESHOLD){
            d_t1_neuron_outputs[neuron] = 1;
            //printf("firing");
        }else{
            d_t1_neuron_outputs[neuron] = 0;
        }
        for(int synapse = 0; synapse < NUM_SYNAPSES_PER_NEURON; synapse++){
            neuron_array[synapse].activity *= ACTIVITY_DISCOUNT_FACTOR;
            neuron_array[synapse].activity += neuron_array[synapse].input * (d_t1_neuron_outputs[neuron] - 0.5);
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

__global__ void read_t1(struct Synapse *d_t1_synapses, size_t t1_pitch){
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if(neuron < NUM_T1_NEURONS){
        struct Synapse *neuron_array = (struct Synapse *) ((char*)d_t1_synapses + neuron * t1_pitch);
        for(int synapse = 0; synapse < NUM_SYNAPSES_PER_NEURON; synapse++){
            neuron_array[synapse].input = (*neuron_array[synapse].p_presynaptic_output);
        }
    }
}


__global__ void learn(struct Synapse *d_synapses, float learning_factor, size_t pitch, int *d_neuron_outputs, int *d_brain_inputs, curandState_t *d_curand_state){
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if(neuron < NUM_NEURONS){
        struct Synapse *neuron_array = (struct Synapse *) ((char*)d_synapses + neuron * pitch);
        for(int synapse = 0; synapse < NUM_SYNAPSES_PER_NEURON; synapse++){
            //neuron_array[synapse].weight += LEARNING_RATE * reward * neuron_array[synapse].activity;
            //neuron_array[synapse].weight += LEARNING_RATE * reward * neuron_array[synapse].activity * fabs(MAX_ABS_WEIGHT - fabs(neuron_array[synapse].weight));
            neuron_array[synapse].weight += LEARNING_RATE * learning_factor * neuron_array[synapse].activity;
            neuron_array[synapse].weight *= pow(2.0, -WEIGHT_VALUE_LIMIT_FACTOR * neuron_array[synapse].weight * neuron_array[synapse].weight);

            //randomly reconnect if weight too small
            if(fabsf(neuron_array[synapse].weight) < MIN_ABS_WEIGHT){
                float new_weight = curand_uniform(&d_curand_state[neuron]) + MIN_START_WEIGHT;
                neuron_array[synapse].weight = new_weight;
                //printf("reconnecting");
                int rand_input = curand(&d_curand_state[neuron]) % (NUM_NEURONS + NUM_INPUTS);
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


__global__ void learn_t1(struct Synapse *d_t1_synapses, float reward, size_t t1_pitch, int *d_t1_neuron_outputs, int *d_neuron_outputs, int *d_brain_inputs, curandState_t *d_t1_curand_states){
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(neuron < NUM_T1_NEURONS){
        struct Synapse *neuron_array = (struct Synapse *) ((char*)d_t1_synapses + neuron * t1_pitch);
        for(int synapse = 0; synapse < NUM_SYNAPSES_PER_NEURON; synapse++){
            //neuron_array[synapse].weight += LEARNING_RATE * reward * neuron_array[synapse].activity;
            //neuron_array[synapse].weight += LEARNING_RATE * reward * neuron_array[synapse].activity * fabs(MAX_ABS_WEIGHT - fabs(neuron_array[synapse].weight));
            neuron_array[synapse].weight += LEARNING_RATE * reward * neuron_array[synapse].activity;
            neuron_array[synapse].weight *= pow(2.0, -WEIGHT_VALUE_LIMIT_FACTOR * neuron_array[synapse].weight * neuron_array[synapse].weight);

            //randomly reconnect if weight too small
            if(fabsf(neuron_array[synapse].weight) < MIN_ABS_WEIGHT){
                int rand_input = curand(&d_t1_curand_states[neuron]) % (NUM_NEURONS + NUM_T1_NEURONS + NUM_INPUTS);
                if(rand_input < NUM_NEURONS){
                    //connect to neuron from main ANN
                    neuron_array[synapse].p_presynaptic_output = &d_neuron_outputs[rand_input];
                }else if(rand_input < NUM_NEURONS + NUM_T1_NEURONS){
                    //connect to T1 neuron
                    neuron_array[synapse].p_presynaptic_output = &d_t1_neuron_outputs[rand_input - NUM_NEURONS];
                }else{
                    //connect to brain input
                    neuron_array[synapse].p_presynaptic_output = &d_brain_inputs[rand_input - (NUM_NEURONS + NUM_T1_NEURONS)];
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


__global__ void update_parameters(struct Parameters *d_parameters){
    d_parameters->threshold_randomness_factor *= 1;
    //printf("new factor %.2f ", d_parameters->threshold_randomness_factor);
}
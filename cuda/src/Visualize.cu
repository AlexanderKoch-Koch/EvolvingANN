#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "Synapse.h"
#include "Hyperparameters.h"
#include "Parameters.h"
#include "TensorboardInterface.h"


__global__ void printSynapses(struct Synapse *d_synapses, size_t pitch){
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if(neuron < NUM_NEURONS){
        float weight_sum = 0;
        struct Synapse *neuron_array = (struct Synapse *) ((char*)d_synapses + neuron * pitch);
        for(int synapse = 0; synapse< NUM_SYNAPSES_PER_NEURON; synapse++){
            printf("neuron: %d, synapse: %d, weight: %.2f, activity: %.2f, input: %d\n",
                neuron, synapse, neuron_array[synapse].weight,
                neuron_array[synapse].activity, neuron_array[synapse].input);
            weight_sum += neuron_array[synapse].weight;
        }
        printf("avr weight: %.2f  ", weight_sum / NUM_SYNAPSES_PER_NEURON);
    }
}

__global__ void print_synapse_stats(struct Synapse *d_synapses, size_t pitch){
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if(neuron < NUM_NEURONS){
        float weight_sum = 0;
        struct Synapse *neuron_array = (struct Synapse *) ((char*)d_synapses + neuron * pitch);
        for(int synapse = 0; synapse< NUM_SYNAPSES_PER_NEURON; synapse++){
            weight_sum += neuron_array[synapse].weight;
        }
        float mean = weight_sum /NUM_SYNAPSES_PER_NEURON;
        printf("avr weight: %.2f  ", mean);
        //compute standard deviation
        float standard_deviation = 0.0;
        for(int synapse = 0; synapse< NUM_SYNAPSES_PER_NEURON; synapse++){
            standard_deviation += (neuron_array[synapse].weight - mean) * (neuron_array[synapse].weight - mean);
        }
        printf("standard_deviation: %.2f   ", standard_deviation);
    }
}

__global__ void printNeurons(int *d_neuron_outputs, float *d_weighted_sums){
    int neuron = blockIdx.x*blockDim.x + threadIdx.x;
    printf("neuron: %d, weighted sum: %.2f, output: %d\n", neuron, d_weighted_sums[neuron], d_neuron_outputs[neuron]);
}


void neuron_stats(int *d_neuron_outputs, unsigned long step){
    int *neuron_outputs = (int*) malloc(sizeof(int) * NUM_NEURONS);
    cudaMemcpy(neuron_outputs, d_neuron_outputs, sizeof(int) * NUM_NEURONS, cudaMemcpyDeviceToHost);
    int output_sum = 0;
    for(int i = 0; i <NUM_NEURONS; i++){
        output_sum += neuron_outputs[i];
    }
    write_scalar(step, output_sum/(float)NUM_NEURONS, "avr_output");
}


void synapse_stats(struct Synapse *d_synapses, size_t pitch, unsigned long step){
    const int num_neurons_to_analyse = 10;
    struct Synapse synapses[num_neurons_to_analyse][NUM_SYNAPSES_PER_NEURON] = {0}; 

    size_t h_pitch = sizeof(struct Synapse) * NUM_SYNAPSES_PER_NEURON;
    cudaMemcpy2D(synapses, h_pitch, d_synapses, pitch, sizeof(struct Synapse) * NUM_SYNAPSES_PER_NEURON, num_neurons_to_analyse, cudaMemcpyDeviceToHost);
    
    double weight_sum = 0;
    double abs_weight_sum = 0;
    double activity_sum = 0;
    for(int neuron = 0; neuron < num_neurons_to_analyse; neuron++){
        for(int synapse = 0; synapse < NUM_SYNAPSES_PER_NEURON; synapse++){
            weight_sum += synapses[neuron][synapse].weight;
            abs_weight_sum += fabsf(synapses[neuron][synapse].weight);
            activity_sum += fabsf(synapses[neuron][synapse].activity);
        }
    }
    write_scalar(step, weight_sum/(float)(NUM_SYNAPSES_PER_NEURON * num_neurons_to_analyse), "avr_weight");
    write_scalar(step, abs_weight_sum/(float)(NUM_SYNAPSES_PER_NEURON * num_neurons_to_analyse), "avr_abs_weight");
    write_scalar(step, activity_sum/ float(NUM_SYNAPSES_PER_NEURON * num_neurons_to_analyse), "avr_abs_activity");
}

__global__ void print_parameters(struct Parameters *d_parameters){
    printf(" threshold_randomness_factor %.3f ", d_parameters->threshold_randomness_factor);   
}
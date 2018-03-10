#include <stdio.h>
#include "Neuron.h"
#include "Parameters.h"
#include "Synapse.h"
#include <math.h>



void compute(struct Synapse **neurons, int num_neurons, int num_synapses_per_neuron, float *neuron_outputs){
  float weighted_sum = 0;
  int num_neurons_fired = 0;
  for(int neuron = 0; neuron < num_neurons; neuron++){
    weighted_sum = 0;
    for(int synapse = 0; synapse < num_synapses_per_neuron; synapse++){
      //Synapse activity discount
      neurons[neuron][synapse].activity *= SYNAPSE_DISCOUNT_FACTOR;
      //Adding neuron inputs
      printf("weight: %.2f, input: %.2f\t", neurons[neuron][synapse].weight, neurons[neuron][synapse].input);
      weighted_sum += neurons[neuron][synapse].weight * neurons[neuron][synapse].input;
      
    }
    if(weighted_sum > THRESHOLD){
       neuron_outputs[neuron] = 1;
    }
    
    for(int synapse = 0; synapse < num_synapses_per_neuron; synapse++){
      neurons[neuron][synapse].activity += 100 * neurons[neuron][synapse].input *neurons[neuron][synapse].weight *  weighted_sum;
    }

    for(int synapse = 0; synapse < num_synapses_per_neuron; synapse++){
      neurons[neuron][synapse].activity += neurons[neuron][synapse].input * neurons[neuron][synapse].weight *  neuron_outputs[neuron];
    }
  }
}


void read(struct Synapse **neurons, int num_neurons, int num_synapses_per_neuron){
  for(int neuron = 0; neuron < num_neurons; neuron++){
    for(int synapse = 0; synapse < num_synapses_per_neuron; synapse++){
      //copying input from presynaptic output into own input
      neurons[neuron][synapse].input = *neurons[neuron][synapse].p_presynaptic_output;
    }
  }
}


void learn(struct Synapse **neurons, int num_neurons, int num_synapses_per_neuron, float reward){
  DEBUG_PRINT(("reward: %f\n", reward));
  for(int neuron = 0; neuron < num_neurons; neuron++){
    for(int synapse = 0; synapse < num_synapses_per_neuron; synapse++){
      float activity = neurons[neuron][synapse].activity;
      //update weights
      float weight_change = LEARNING_RATE * activity * reward / fabsf(neurons[neuron][synapse].weight);
      neurons[neuron][synapse].weight += weight_change;
      #ifdef DEBUG
      printf("activity: %.2f weight_change%.4f new_weight:%.2f  \t", activity, weight_change, neurons[neuron][synapse].weight);
      #endif
    }
  }
}


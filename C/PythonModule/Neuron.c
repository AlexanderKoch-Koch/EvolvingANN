#include <stdio.h>
#include "Neuron.h"
#include "Parameters.h"
#include "Synapse.h"


void compute(struct Synapse **neurons, int num_neurons, int num_synapses_per_neuron, int *neuron_outputs){
  int weighted_sum = 0;
  for(int neuron = 0; neuron < num_neurons; neuron++){
    weighted_sum = 0;
    for(int synapse = 0; synapse < num_synapses_per_neuron; synapse++){
      neurons[neuron][synapse].activity *= SYNAPSE_DISCOUNT_FACTOR;
      printf("input %d ", neurons[neuron][synapse].input);
      if(neurons[neuron][synapse].input == 1){
        weighted_sum += neurons[neuron][synapse].weight;
      }
    }
    if(weighted_sum > THRESHOLD){
       neuron_outputs[neuron] = 1;
    }
    else{
      printf("not fire\n");
       neuron_outputs[neuron] = 0;
    }

    for(int synapse = 0; synapse < num_synapses_per_neuron; synapse++){
      neurons[neuron][synapse].activity += neurons[neuron][synapse].input * neurons[neuron][synapse].weight *  neuron_outputs[neuron];
    }
  }
}


void read(struct Synapse **neurons, int num_neurons, int num_synapses_per_neuron){
  for(int neuron = 0; neuron < num_neurons; neuron++){
    for(int synapse = 0; synapse < num_synapses_per_neuron; synapse++){
      neurons[neuron][synapse].input = *neurons[neuron][synapse].p_presynaptic_output;
    }
  }
}


void learn(struct Synapse **neurons, int num_neurons, int num_synapses_per_neuron, float reward){
  for(int neuron = 0; neuron < num_neurons; neuron++){
    for(int synapse = 0; synapse < num_synapses_per_neuron; synapse++){
      neurons[neuron][synapse].weight += LEARNING_RATE * neurons[neuron][synapse].activity * reward;
    }
  }
}

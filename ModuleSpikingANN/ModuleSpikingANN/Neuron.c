#include <stdio.h>
#include "Neuron.h"
#include "Parameters.h"
#include "Synapse.h"
#include <math.h>



void compute(struct Synapse **neurons, int num_neurons, int num_synapses_per_neuron, int *neuron_outputs){
  float weighted_sum = 0;
  int num_neurons_fired = 0;
  for(int neuron = 0; neuron < num_neurons; neuron++){
    weighted_sum = 0;
    for(int synapse = 0; synapse < num_synapses_per_neuron; synapse++){
      //Synapse activity discount
      neurons[neuron][synapse].activity *= SYNAPSE_DISCOUNT_FACTOR;
      //Adding neuron inputs
      if(neurons[neuron][synapse].input == 1){
        weighted_sum += neurons[neuron][synapse].weight;
      }
    }
    if(weighted_sum > THRESHOLD){
       num_neurons_fired += 1;
       neuron_outputs[neuron] = 1;
       for(int synapse = 0; synapse < num_synapses_per_neuron; synapse++){
         int weight_sign = neurons[neuron][synapse].weight / neurons[neuron][synapse].weight;
         DEBUG_PRINT(("weight: %f sign %d ", neurons[neuron][synapse].weight, weight_sign));
         //Updating synapse activity
         neurons[neuron][synapse].activity += neurons[neuron][synapse].input * weight_sign;
       }
    }
    else{
       neuron_outputs[neuron] = 0;
    }
  }
  DEBUG_PRINT(("%d of %d neurons fired\n", num_neurons_fired, num_neurons));
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
    #ifdef DEBUG
    printf("\nNeuron %d weights:", neuron);
    #endif
    for(int synapse = 0; synapse < num_synapses_per_neuron; synapse++){
      float activity = neurons[neuron][synapse].activity;
      //update weights
      neurons[neuron][synapse].weight += LEARNING_RATE * activity * reward / fabsf(neurons[neuron][synapse].weight);
      #ifdef DEBUG
      printf("%.2f ", neurons[neuron][synapse].weight);
      #endif
    }
  }
}

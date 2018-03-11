#include <stdio.h>
#include "Neuron.h"
#include "Parameters.h"
#include "Synapse.h"
#include <math.h>



void compute(struct Synapse **neurons, int num_neurons, int num_synapses_per_neuron, int *neuron_outputs){
  float weighted_sum = 0;
  float sum_outputs = 0;
  int num_neurons_fired = 0;
  for(int neuron = 0; neuron < num_neurons; neuron++){
    weighted_sum = 0;
    for(int synapse = 0; synapse < num_synapses_per_neuron; synapse++){
      //Synapse activity discount
      neurons[neuron][synapse].activity *= SYNAPSE_DISCOUNT_FACTOR;
      //Adding neuron inputs
      //printf("weight: %.2f, input: %d\t", neurons[neuron][synapse].weight, neurons[neuron][synapse].input);
      weighted_sum += neurons[neuron][synapse].weight * neurons[neuron][synapse].input;
      
    }

    if(weighted_sum > THRESHOLD){
      neuron_outputs[neuron] = 1;
    }else{
      neuron_outputs[neuron] = 0;
    }
    sum_outputs += neuron_outputs[neuron];
    for(int synapse = 0; synapse < num_synapses_per_neuron; synapse++){
      neurons[neuron][synapse].activity += neurons[neuron][synapse].input * neurons[neuron][synapse].weight *  neuron_outputs[neuron];
      //printf("weight: %.2f, input: %d, neuron_outputs: %d, activity: %.2f\t", neurons[neuron][synapse].weight, neurons[neuron][synapse].input, neuron_outputs[neuron], neurons[neuron][synapse].activity);
    }
  }
  printf("\navr_output: %.2f", sum_outputs/num_neurons);
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
  double sum_weights = 0;
  double sum_activities = 0;
  for(int neuron = 0; neuron < num_neurons; neuron++){
    for(int synapse = 0; synapse < num_synapses_per_neuron; synapse++){
      sum_activities += neurons[neuron][synapse].activity;
      sum_weights += neurons[neuron][synapse].weight;
      //printf("weight: %.2f, ", neurons[neuron][synapse].weight);
      float activity = neurons[neuron][synapse].activity;
      float weight = neurons[neuron][synapse].weight;
      if(weight != 0){
        //update weights
        float weight_change = LEARNING_RATE * activity * reward * fabs(THRESHOLD + 0.5 - fabs(weight));
        //float weight_change = LEARNING_RATE * activity * reward;
        neurons[neuron][synapse].weight += weight_change;
        printf("activity: %.2f weight_change%.4f new_weight:%.2f  \t", activity, weight_change, neurons[neuron][synapse].weight);
      }
    }
  }
  int num_synapses = num_neurons * num_synapses_per_neuron;
  printf("\navr_activity: %.2f, avr_weight: %.2f", sum_activities/num_synapses, sum_weights/num_synapses);
}


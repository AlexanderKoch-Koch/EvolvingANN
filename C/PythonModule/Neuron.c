#include <stdio.h>
#include "Neuron.h"
#include "Parameters.h"
#include "Synapse.h"
#include <math.h>



void compute(struct Synapse **synapses, int *neuron_outputs, struct Parameters *p){
  float weighted_sum = 0;
  float sum_outputs = 0;
  int num_neurons_fired = 0;

  for(int neuron = 0; neuron < p->num_neurons; neuron++){
    weighted_sum = 0;
    for(int synapse = 0; synapse < p->num_synapses_per_neuron; synapse++){
      synapses[neuron][synapse].activity *= p->activity_discount_factor;
      //printf("weight: %.2f, input: %d\t", synapses[neuron][synapse].weight, synapses[neuron][synapse].input);
      weighted_sum += synapses[neuron][synapse].weight * synapses[neuron][synapse].input; 
    }

    neuron_outputs[neuron] = (weighted_sum > p->threshold) ? 1 : 0;
    sum_outputs += neuron_outputs[neuron];

    for(int synapse = 0; synapse < p->num_synapses_per_neuron; synapse++){
      synapses[neuron][synapse].activity += synapses[neuron][synapse].input * synapses[neuron][synapse].weight *  neuron_outputs[neuron];
      //printf("weight: %.2f, input: %d, neuron_outputs: %d, activity: %.2f\t", synapses[neuron][synapse].weight, synapses[neuron][synapse].input, neuron_outputs[neuron], neurons[neuron][synapse].activity);
    }
  }
  DEBUG_PRINT(("\navr_output: %.2f", sum_outputs/p->num_neurons));
}


void read(struct Synapse **synapses, struct Parameters *p){
  for(int neuron = 0; neuron < p->num_neurons; neuron++){
    for(int synapse = 0; synapse < p->num_synapses_per_neuron; synapse++){
      //copying input from presynaptic output into own input
      synapses[neuron][synapse].input = *synapses[neuron][synapse].p_presynaptic_output;
    }
  }
}


void learn(struct Synapse **synapses, float reward, struct Parameters *p){
  double sum_weights = 0;
  double sum_activities = 0;

  for(int neuron = 0; neuron < p->num_neurons; neuron++){
    for(int synapse = 0; synapse < p->num_synapses_per_neuron; synapse++){
      sum_activities += synapses[neuron][synapse].activity;
      sum_weights += synapses[neuron][synapse].weight;

      //printf("weight: %.2f, ", synapses[neuron][synapse].weight);
      float activity = synapses[neuron][synapse].activity;
      float weight = synapses[neuron][synapse].weight;
      //update weights
      float weight_change = p->learning_rate * activity * reward * fabs(p->max_weight_value - fabs(weight));
      //float weight_change = LEARNING_RATE * activity * reward;
      synapses[neuron][synapse].weight += weight_change;
      DEBUG_PRINT(("activity: %.2f weight_change%.4f new_weight:%.2f  \t", activity, weight_change, synapses[neuron][synapse].weight));
    }
  }
  int num_synapses = p->num_neurons * p->num_synapses_per_neuron;
  DEBUG_PRINT(("\navr_activity: %.2f, avr_weight: %.2f", sum_activities/num_synapses, sum_weights/num_synapses));
}


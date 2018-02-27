#include <stdio.h>
#include "Neuron.h"
#include "Parameters.h"



void compute(int inputs[NUM_NEURONS][NUM_SYNAPSES_PER_NEURON],
             float weights[NUM_NEURONS][NUM_SYNAPSES_PER_NEURON],
             int neuron_outputs[NUM_NEURONS]){
  int weighted_sum = 0;
  for(int neuron = 0; neuron < NUM_NEURONS; neuron++){
    weighted_sum = 0;
    for(int input = 0; input < NUM_SYNAPSES_PER_NEURON; input++){
      if(inputs[neuron][input] == 1){
        weighted_sum += weights[neuron][input];
      }
    }
    if(weighted_sum > THRESHOLD){
       neuron_outputs[neuron] = 1;
       printf("fire\n");
    }
    else neuron_outputs[neuron] = 0;
  }
}


void read(int inputs[NUM_NEURONS][NUM_SYNAPSES_PER_NEURON], int *p_presynaptic_neuron_outputs[NUM_NEURONS][NUM_SYNAPSES_PER_NEURON]){
  for(int neuron = 0; neuron < NUM_NEURONS; neuron++){
    for(int synapse = 0; synapse < NUM_SYNAPSES_PER_NEURON; synapse++){
      inputs[neuron][synapse] = *p_presynaptic_neuron_outputs[neuron][synapse];
    }
  }
}


void learn(float weights[NUM_NEURONS][NUM_SYNAPSES_PER_NEURON],
           float synapse_activities[NUM_NEURONS][NUM_SYNAPSES_PER_NEURON],
           float reward){
  for(int neuron = 0; neuron < NUM_NEURONS; neuron++){
    for(int synapse = 0; synapse < NUM_SYNAPSES_PER_NEURON; synapse++){
      weights[neuron][synapse] += LEARNING_RATE * synapse_activities[neuron][synapse] * reward;
    }
  }
}


void tag_synapse(float synapse_activities[NUM_NEURONS][NUM_SYNAPSES_PER_NEURON],
  int neuron_outputs[NUM_NEURONS],
  int neuron_inputs[NUM_NEURONS][NUM_SYNAPSES_PER_NEURON]){
    for(int neuron = 0; neuron < NUM_NEURONS; neuron++){
      for(int synapse = 0; synapse < NUM_SYNAPSES_PER_NEURON; synapse++){
        synapse_activities[neuron][synapse] *= SYNAPSE_DISCOUNT_FACTOR;
        synapse_activities[neuron][synapse] += neuron_inputs[neuron][synapse] * neuron_outputs[neuron];
      }
    }
}

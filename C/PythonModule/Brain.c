#include "Brain.h"
#include <stdio.h>
#include <string.h>
#include "Neuron.h"
#include "Parameters.h"

float weights[NUM_NEURONS][NUM_SYNAPSES_PER_NEURON];
float synapse_activities[NUM_NEURONS][NUM_SYNAPSES_PER_NEURON] = {{0}};
int neuron_inputs[NUM_NEURONS][NUM_SYNAPSES_PER_NEURON] = {{0}};
int neuron_outputs[NUM_NEURONS] = {0};
int *p_presynaptic_neuron_outputs[NUM_NEURONS][NUM_SYNAPSES_PER_NEURON];
int brain_inputs[NUM_INPUTS] = {0};
int zero = 0;
int num_calls = 0;
int num_neurons, num_inputs, num_outputs;

void init_brain(int num_neurons_p, int num_inputs_p, int num_outputs_p){
  num_neurons = num_neurons_p;
  num_inputs = num_inputs_p;
  num_outputs = num_outputs_p;
  num_calls += 1;
  for(int neuron = 0; neuron < NUM_NEURONS; neuron++){
    for(int synapse = 0; synapse < NUM_SYNAPSES_PER_NEURON; synapse++){
      weights[neuron][synapse] = 1.5;
      p_presynaptic_neuron_outputs[neuron][synapse] = &zero;
    }
  }
  //wiring
  p_presynaptic_neuron_outputs[0][1] = &brain_inputs[0];
  p_presynaptic_neuron_outputs[1][1] = &brain_inputs[1];
  p_presynaptic_neuron_outputs[2][1] = &neuron_outputs[3];
  p_presynaptic_neuron_outputs[3][1] = &neuron_outputs[4];
  p_presynaptic_neuron_outputs[4][1] = &neuron_outputs[0];
  p_presynaptic_neuron_outputs[0][0] = &brain_inputs[1];
  p_presynaptic_neuron_outputs[1][0] = &neuron_outputs[2];
  p_presynaptic_neuron_outputs[2][0] = &neuron_outputs[3];
  p_presynaptic_neuron_outputs[3][0] = &neuron_outputs[4];
  p_presynaptic_neuron_outputs[4][0] = &neuron_outputs[0];
}

void think(int *inputs){
  //copy inputs to brain inputs
  memcpy(brain_inputs, inputs, 2 * sizeof(int));
  //save inputs in connected neurons
  read(neuron_inputs, p_presynaptic_neuron_outputs);
  //reset brain_inputs to 0
  memset(brain_inputs, 0, NUM_INPUTS * sizeof(int));

  for(int think_step = 0; think_step < 2; think_step++){
    compute(neuron_inputs, weights, neuron_outputs);
    tag_synapse(synapse_activities, neuron_outputs, neuron_inputs);
    read(neuron_inputs, p_presynaptic_neuron_outputs);
  }
}

void process_reward(float reward){
  learn(weights, synapse_activities, reward);
}

int get_num(){
  return num_calls;
}

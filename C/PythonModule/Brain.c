#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "Brain.h"
#include "Neuron.h"
#include "Parameters.h"
#include "Synapse.h"


//all neurons
struct Synapse **neurons;
float *brain_inputs;
float *neuron_outputs;
int num_neurons = -1;
int num_inputs = -1;
int output_index_max = -1;
int num_synapses_per_neuron;
int zero = 0;

float rand_range(float min_n, float max_n)
{
    return (float)rand()/RAND_MAX * (max_n - min_n) + min_n;
}

void init_brain(int num_neurons_p, int num_inputs_p, int num_outputs_p){
  num_neurons = num_neurons_p + num_outputs_p;
  num_inputs = num_inputs_p;
  output_index_max = num_outputs_p - 1; //output neurons get indexes 0 to num_outputs - 1
  num_synapses_per_neuron = num_inputs_p + num_neurons;

  //allocate memory for ANN values
  brain_inputs = (float*) calloc(num_inputs, sizeof(float));
  neuron_outputs = (float*) calloc(num_neurons, sizeof(float));

  /* Intializes random number generator */
  time_t t;   
  srand((unsigned) time(&t));

  neurons = (struct Synapse**) malloc(sizeof(struct Synapse) * num_neurons);
  for(int neuron = 0; neuron < num_neurons; neuron++){
    //allocate memory for synapse array of each neuron and set all values to 0
    neurons[neuron] = (struct Synapse*) calloc(num_synapses_per_neuron, sizeof(struct Synapse));
    for(int synapse = 0; synapse < num_synapses_per_neuron; synapse++){
      neurons[neuron][synapse].weight = rand_range(-0.4, 1);
      //connect randomly to a neuron/input
      int new_presynaptic_neuron = rand() % (num_neurons + num_inputs);
      if(new_presynaptic_neuron >= num_neurons){
        //connect to input new_presynaptic_neuron - num_neurons
        neurons[neuron][synapse].p_presynaptic_output = &brain_inputs[new_presynaptic_neuron - num_neurons];
      }else{
        neurons[neuron][synapse].p_presynaptic_output = &neuron_outputs[new_presynaptic_neuron];
      }
    }
  }
}

int * think(float *inputs, int len_inputs, int *num_outputs){
  if(num_neurons < 0){
    printf("You have to call init before using other functions");
    return &zero;
  }
  if(len_inputs != num_inputs){
    printf("list length is wrong");
    return &zero;
  }
  *num_outputs = output_index_max + 1;
  //copy inputs to brain inputs
  memcpy(brain_inputs, inputs, num_inputs * sizeof(float));
  //save inputs in connected neurons
  read(neurons, num_neurons, num_synapses_per_neuron);
  //reset brain_inputs to 0
  memset(brain_inputs, 0, num_inputs * sizeof(float));
  compute(neurons, num_neurons, num_synapses_per_neuron, neuron_outputs);
  float *brain_outputs = (float*) malloc(sizeof(float) * (*num_outputs));
  memcpy(brain_outputs, neuron_outputs, *num_outputs * sizeof(float));

  #ifdef DEBUG
  printf("num neurons: %d ", num_neurons);
  printf("neuron_outputs[0]: %f", neuron_outputs[0]);
  printf("\n");
  #endif

  return brain_outputs;
}

void process_reward(float reward){
  if(num_neurons < 0){
    printf("You have to call init before using other functions");
    return;
  }
  learn(neurons, num_neurons, num_synapses_per_neuron, reward);

  //reconnect
  for(int neuron = 0; neuron < num_neurons; neuron++){
    for(int synapse = 0; synapse < num_synapses_per_neuron; synapse++){
      //printf("weight %f", fabsf(neurons[neuron][synapse].weight));
      if(fabsf(neurons[neuron][synapse].weight) < 0.1){
        printf("reconnecting neuron %d synapse %d\n", neuron, synapse);
        //reconnect randomly
        neurons[neuron][synapse].weight = rand_range(-0.5, 0.6);
        int new_presynaptic_neuron = rand() % (num_neurons + num_inputs);
        if(new_presynaptic_neuron >= num_neurons){
          //connect to input new_presynaptic_neuron - num_neurons
          neurons[neuron][synapse].p_presynaptic_output = &brain_inputs[new_presynaptic_neuron - num_neurons];
        }else{
          neurons[neuron][synapse].p_presynaptic_output = &neuron_outputs[new_presynaptic_neuron];
        }
      }
    }
  }
}

void reset_memory(){
  for(int neuron = 0; neuron < num_neurons; neuron++){
    neuron_outputs[neuron] = 0;
    for(int synapse = 0; synapse < num_synapses_per_neuron; synapse++){
      //synapse activity reset
      neurons[neuron][synapse].activity = 0;
      neurons[neuron][synapse].input = 0;
    }
  }
}


void release_memory(){
  if(num_neurons < 0){
    printf("You have to call init before using other functions");
    return;
  }
  for(int neuron = 0; neuron < num_neurons; neuron++){
    free(neurons[neuron]);
  }
  free(neurons);
  free(brain_inputs);
  num_neurons = -1;
}

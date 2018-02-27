#include "Brain.h"
#include <stdio.h>
#include <string.h>
#include "Neuron.h"
#include "Parameters.h"
#include <stdlib.h>
#include "Synapse.h"

struct Synapse **neurons;
int *brain_inputs;
int *neuron_outputs;
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
  printf("%d neurons %d inputs %d outputs\n", num_neurons_p, num_inputs_p, num_outputs_p);
  num_neurons = num_neurons_p + num_outputs_p;
  num_inputs = num_inputs_p;
  output_index_max = num_outputs_p - 1; //output neurons get indexes 0 to num_outputs - 1
  num_synapses_per_neuron = num_neurons + num_inputs;


  //allocate memory for ANN values
  brain_inputs = (int*) calloc(num_inputs, sizeof(int));
  neuron_outputs = (int*) calloc(num_neurons, sizeof(int));
  neurons = (struct Synapse**) malloc(sizeof(struct Synapse) * num_neurons);
  for(int neuron = 0; neuron < num_neurons; neuron++){
    //allocate memory for synapse array of each neuron and set all values to 0
    neurons[neuron] = (struct Synapse*) calloc(num_synapses_per_neuron, sizeof(struct Synapse));
    for(int synapse = 0; synapse < num_synapses_per_neuron; synapse++){
      neurons[neuron][synapse].weight = rand_range(0.5, 1.5);
      if(synapse < num_neurons){
        //connect to all other neurons
        neurons[neuron][synapse].p_presynaptic_output = &neuron_outputs[synapse];
        printf("connecting to neuron %d \n", synapse);
      }else{
        //connect to input neurons
        printf("connecting to input %d \n", synapse - num_neurons);
        neurons[neuron][synapse].p_presynaptic_output = &brain_inputs[synapse - num_neurons];
      }
    }
  }
}

int * think(int *inputs, int len_inputs, int *num_outputs){
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
  memcpy(brain_inputs, inputs, num_inputs * sizeof(int));
  //save inputs in connected neurons
  read(neurons, num_neurons, num_synapses_per_neuron);
  //reset brain_inputs to 0
  memset(brain_inputs, 0, num_inputs * sizeof(int));
  compute(neurons, num_neurons, num_synapses_per_neuron, neuron_outputs);

  int *brain_outputs = (int*) malloc(sizeof(int) * output_index_max + 1);
  for(int neuron = 0; neuron < output_index_max; neuron++){
    brain_outputs[neuron] = neuron_outputs[neuron];
  }
  return neuron_outputs;
}

void process_reward(float reward){
  if(num_neurons < 0){
    printf("You have to call init before using other functions");
    return;
  }
  learn(neurons, num_neurons, num_synapses_per_neuron, reward);
}

void reset_memory(){
  for(int neuron = 0; neuron < num_neurons; neuron++){
    for(int synapse = 0; synapse < num_synapses_per_neuron; synapse++){
      neurons[neuron][synapse].activity = 0;
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

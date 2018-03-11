#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "Brain.h"
#include "Neuron.h"
#include "Parameters.h"
#include "Synapse.h"


//all neurons
struct Synapse **synapses;
int *brain_inputs;
int *neuron_outputs;
struct Parameters p;

float rand_range(float min_n, float max_n)
{
    return (float)rand()/RAND_MAX * (max_n - min_n) + min_n;
}

void init_brain(struct Parameters parameters){
  p = parameters;

  //allocate memory
  brain_inputs = (int*) calloc(p.num_inputs, sizeof(int));
  neuron_outputs = (int*) calloc(p.num_neurons, sizeof(int));
  synapses = (struct Synapse**) malloc(sizeof(struct Synapse) * p.num_neurons);

  for(int neuron = 0; neuron < p.num_neurons; neuron++){
    //allocate memory for synapse array of each neuron and set all values to 0
    synapses[neuron] = (struct Synapse*) calloc(p.num_synapses_per_neuron, sizeof(struct Synapse));
    for(int synapse = 0; synapse < p.num_synapses_per_neuron; synapse++){
      synapses[neuron][synapse].weight = rand_range(-p.max_start_weight_sum, p.max_start_weight_sum) / p.num_synapses_per_neuron;
      //connect randomly to a neuron/input
      int new_presynaptic_neuron = rand() % (p.num_neurons + p.num_inputs);
      if(new_presynaptic_neuron >= p.num_neurons){
        //connect to input new_presynaptic_neuron - num_neurons
        synapses[neuron][synapse].p_presynaptic_output = &brain_inputs[new_presynaptic_neuron - p.num_neurons];
      }else{
        synapses[neuron][synapse].p_presynaptic_output = &neuron_outputs[new_presynaptic_neuron];
      }
    }
  }
}

int * think(float *inputs, int len_inputs, int *num_outputs){
  if(len_inputs != p.num_inputs){
    printf("list length is wrong");
    return NULL;
  }
  DEBUG_PRINT(("inputs: "));
  for(int input = 0; input < len_inputs; input++){
    DEBUG_PRINT(("%.2f, ", inputs[input]));
  }
  *num_outputs = p.num_outputs;
  //copy inputs to brain inputs
  for(int input = 0; input < p.num_inputs; input++){
    brain_inputs[input] = (int) (inputs[input] + 0.5);
  }
  //save inputs in connected neurons
  read(synapses, &p);
  //reset brain_inputs to 0
  memset(brain_inputs, 0, p.num_inputs * sizeof(int));
  compute(synapses, neuron_outputs, &p);
  int *brain_outputs = (int*) malloc(sizeof(int) * (p.num_outputs));
  memcpy(brain_outputs, neuron_outputs, p.num_outputs * sizeof(int));

  //printf("num neurons: %d ", num_neurons);
  //printf("neuron_outputs[0]: %d", neuron_outputs[0]);

  return brain_outputs;
}

void process_reward(float reward){
  if(p.num_neurons < 0){
    printf("You have to call init before using other functions");
    return;
  }
  //printf("num_synapses_per_neuron: %d\n", num_synapses_per_neuron);
  learn(synapses, reward, &p);

  //reconnect
  for(int neuron = 0; neuron < p.num_neurons; neuron++){
    for(int synapse = 0; synapse < p.num_synapses_per_neuron; synapse++){
      //printf("weight %f", fabsf(neurons[neuron][synapse].weight));
      if(fabsf(synapses[neuron][synapse].weight) < p.min_weight){
        //printf("reconnecting neuron %d synapse %d\n", neuron, synapse);
        //reconnect randomly
        synapses[neuron][synapse].weight = rand_range(-p.max_start_weight_sum, p.max_start_weight_sum) / p.num_synapses_per_neuron;
        //printf("new_weight_reconn: %f\n", synapses[neuron][synapse].weight);
        int new_presynaptic_neuron = rand() % (p.num_neurons + p.num_inputs);
        if(new_presynaptic_neuron >= p.num_neurons){
          //connect to input new_presynaptic_neuron - num_neurons
          synapses[neuron][synapse].p_presynaptic_output = &brain_inputs[new_presynaptic_neuron - p.num_neurons];
        }else{
          synapses[neuron][synapse].p_presynaptic_output = &neuron_outputs[new_presynaptic_neuron];
        }
      }
    }
  }
}

void reset_memory(){
  for(int neuron = 0; neuron < p.num_neurons; neuron++){
    neuron_outputs[neuron] = 0;
    for(int synapse = 0; synapse < p.num_synapses_per_neuron; synapse++){
      //synapse activity reset
      synapses[neuron][synapse].activity = 0;
      synapses[neuron][synapse].input = 0;
    }
  }
}


void release_memory(){
  if(p.num_neurons < 0){
    printf("You have to call init before using other functions");
    return;
  }
  for(int neuron = 0; neuron < p.num_neurons; neuron++){
    free(synapses[neuron]);
  }
  free(synapses);
  free(brain_inputs);
  p.num_neurons = -1;
}

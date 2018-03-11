#define LEARNING_RATE 0.5
#define THRESHOLD 0
#define NUM_SYNAPSES_PER_NEURON 7
#define SYNAPSE_DISCOUNT_FACTOR 0.8
#define DEBUG


#ifdef DEBUG
# define DEBUG_PRINT(x) printf x
#else
# define DEBUG_PRINT(x) do {} while (0)
#endif

#ifndef SYNAPSE_H
#define SYNAPSE_H


struct Parameters{
  int num_inputs;
  int num_neurons;
  int num_outputs;
  int num_synapses_per_neuron;
  float learning_rate;
  float threshold;
  float activity_discount_factor;
};


#endif

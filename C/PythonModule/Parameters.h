#define DEBUG


#ifdef DEBUG
# define DEBUG_PRINT(x) printf x
#else
# define DEBUG_PRINT(x) do {} while (0)
#endif

#ifndef Parameters_H
#define Parameters_H


 struct Parameters{
  int num_inputs;
  int num_neurons;
  int num_outputs;
  int num_synapses_per_neuron;
  float learning_rate;
  float threshold;
  float activity_discount_factor;
  float max_weight_value;
  float max_start_weight_sum;
  float min_weight;
};


#endif

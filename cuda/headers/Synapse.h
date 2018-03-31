#ifndef SYNAPSE_H
#define SYNAPSE_H

struct Synapse{
  float weight;
  float activity;
  int input;
  int *p_presynaptic_output;
};

#endif
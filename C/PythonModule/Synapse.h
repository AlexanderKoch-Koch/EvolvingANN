#ifndef SYNAPSE_H
#define SYNAPSE_H


struct Synapse{
  float weight;
  float activity;
  float input;
  float *p_presynaptic_output;
};


#endif

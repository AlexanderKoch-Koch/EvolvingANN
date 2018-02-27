#ifndef BRAIN_H
#define BRAIN_H

void init_brain(int num_neurons, int num_inputs, int num_outputs);

void think(int *inputs);

void process_reward(float reward);

int get_num();

#endif

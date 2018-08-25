#ifndef BRAIN_H
#define BRAIN_H

void init();

int* think(int *inputs);

void process_reward(float reward);

void reset_memory();

void release_memory();

void write_tensorboard();

#endif
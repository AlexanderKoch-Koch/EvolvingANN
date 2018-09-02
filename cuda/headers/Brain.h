#ifndef BRAIN_H
#define BRAIN_H

int init(const char *log_dir);

int* think(int *inputs);

void process_reward(float reward);

void reset_memory();

void release_memory();

void write_tensorboard();

#endif
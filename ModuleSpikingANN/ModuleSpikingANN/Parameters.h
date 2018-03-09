#define LEARNING_RATE 0.02
#define THRESHOLD 0
#define NUM_SYNAPSES_PER_NEURON 4
#define SYNAPSE_DISCOUNT_FACTOR 0.75
#define DEBUG


#ifdef DEBUG
# define DEBUG_PRINT(x) printf x
#else
# define DEBUG_PRINT(x) do {} while (0)
#endif

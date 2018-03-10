#define LEARNING_RATE 0.1
#define THRESHOLD 0
#define NUM_SYNAPSES_PER_NEURON 7
#define SYNAPSE_DISCOUNT_FACTOR 0.6
#define DEBUG


#ifdef DEBUG
# define DEBUG_PRINT(x) printf x
#else
# define DEBUG_PRINT(x) do {} while (0)
#endif

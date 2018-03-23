#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define NUM_NEURONS 10
#define NUM_SYNAPSES_PER_NEURON 3

struct Synapse{
  float weight;
  float activity;
  int input;
  int *p_presynaptic_output;
};

__global__ void init_random_seed(unsigned int seed, curandState_t d_curand_state) {

  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              blockIdx.x, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &d_curand_state);
}

__global__ void init(struct Synapse *d_synapses, size_t pitch, curandState_t d_curand_state){
    int synapse = blockIdx.x*blockDim.x + threadIdx.x;
    int neuron = blockIdx.y*blockDim.y + threadIdx.y;

    skipahead(synapse * (neuron + 1), &d_curand_state);
    float new_weight = curand_uniform(&d_curand_state);
    
    struct Synapse *row_a = (struct Synapse *) ((char*)d_synapses + neuron * pitch);
    row_a[synapse].weight = new_weight;
    
    printf("neuron: %d, synapse: %d, new_weight: %.2f\n", neuron, synapse,  new_weight);
}

__global__ void compute(struct Synapse *d_synapses, int *d_neuron_outputs, size_t pitch){
    int synapse = blockIdx.x*blockDim.x + threadIdx.x;
    int neuron = blockIdx.y*blockDim.y + threadIdx.y;

    struct Synapse *row_a = (struct Synapse *) ((char*)d_synapses + neuron * pitch);
}

__global__ void printSynapses(struct Synapse *d_synapses, size_t pitch){
    int synapse = blockIdx.x*blockDim.x + threadIdx.x;
    int neuron = blockIdx.y*blockDim.y + threadIdx.y;

    struct Synapse *row_a = (struct Synapse *) ((char*)d_synapses + neuron * pitch);
    printf("neuron: %d, synapse: %d, weight: %.2f, activity: %.2f, input: %d\n", neuron, synapse, row_a[synapse].weight, row_a[synapse].activity, row_a[synapse].input);
}

__global__ void printNeurons(int *d_neuron_outputs){
    int neuron = blockIdx.x*blockDim.x + threadIdx.x;
    printf("neuron: %d, output: %d\n", neuron, d_neuron_outputs[neuron]);
}

int main(void){
    curandState_t d_curand_state;
    cudaMalloc((void**) &d_curand_state, sizeof(curandState_t));
    init_random_seed<<<1, 1>>>(time(0), d_curand_state);
  
    size_t dev_pitch;
    struct Synapse *d_synapses;
    int *d_neuron_outputs;
    cudaMalloc(&d_neuron_outputs, sizeof(int) * NUM_NEURONS);
    cudaMallocPitch(&d_synapses, &dev_pitch, NUM_SYNAPSES_PER_NEURON * sizeof(struct Synapse), NUM_NEURONS);
    
    dim3 synapses_dim(NUM_SYNAPSES_PER_NEURON, NUM_NEURONS, 1);

    init<<<1, synapses_dim>>>(d_synapses, dev_pitch, d_curand_state);
    cudaDeviceSynchronize();
    //compute<<<1, synapses_dim>>>(d_synapses, d_neuron_outputs, dev_pitch);
    cudaDeviceSynchronize();
    printSynapses<<<1, synapses_dim>>>(d_synapses, dev_pitch);
    printNeurons<<<1, NUM_NEURONS>>>(d_neuron_outputs);
    cudaDeviceSynchronize();
    return 0;
}
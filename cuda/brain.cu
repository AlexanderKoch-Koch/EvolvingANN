#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>

#define NUM_NEURONS 1000
#define NUM_INPUTS 4
#define NUM_OUTPUTS 3
#define NUM_SYNAPSES_PER_NEURON 1000
#define THRESHOLD 0

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

__global__ void init_synapses(struct Synapse *d_synapses, size_t pitch, int *d_neuron_outputs, int *d_brain_inputs, curandState_t d_curand_state){
    int synapse = blockIdx.x*blockDim.x + threadIdx.x;
    int neuron = blockIdx.y*blockDim.y + threadIdx.y;

    skipahead(synapse * (neuron + 1), &d_curand_state);
    float new_weight = curand_uniform(&d_curand_state);
    
    struct Synapse *neuron_array = (struct Synapse *) ((char*)d_synapses + neuron * pitch);
    neuron_array[synapse].weight = new_weight;
    
    int rand_input = curand(&d_curand_state) % (NUM_NEURONS + NUM_INPUTS);
    if(rand_input < NUM_NEURONS){
        //connect to other neuron
        neuron_array[synapse].p_presynaptic_output = &d_neuron_outputs[rand_input];
    }else{
        //connect to brain input
        neuron_array[synapse].p_presynaptic_output = &d_brain_inputs[rand_input - NUM_NEURONS];
    }
    //printf("neuron: %d, synapse: %d, new_weight: %.2f\n", neuron, synapse,  new_weight);
}

__global__ void init_neurons(int *d_neuron_outputs, float *d_weighted_sums){
    int neuron = blockIdx.x*blockDim.x + threadIdx.x;
    
    d_neuron_outputs[neuron] = 0;
    d_weighted_sums[neuron] = 0;
}

__global__ void compute_synapses(struct Synapse *d_synapses, float *d_weighted_sums, size_t pitch){
    int synapse = blockIdx.x*blockDim.x + threadIdx.x;
    int neuron = blockIdx.y*blockDim.y + threadIdx.y;

    struct Synapse *neuron_array = (struct Synapse *) ((char*)d_synapses + neuron * pitch);
    float sum = neuron_array[synapse].weight *  (*neuron_array[synapse].p_presynaptic_output);
    atomicAdd(&d_weighted_sums[neuron], sum);
    //printf("neuron: %d, synapse: %d,  adding %d\n", neuron, synapse, *neuron_array[synapse].p_presynaptic_output);
}

__global__ void compute_neurons(int *d_neuron_outputs, float *d_weighted_sums){
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(d_weighted_sums[neuron] >= THRESHOLD){
        d_neuron_outputs[neuron] = 1;
        printf("firing :)");
    }else{
        d_neuron_outputs[neuron] = 0;
    }
    
    //reset weighted sum
    d_weighted_sums[neuron] = 0.0;
}

__global__ void printSynapses(struct Synapse *d_synapses, size_t pitch){
    int synapse = blockIdx.x*blockDim.x + threadIdx.x;
    int neuron = blockIdx.y*blockDim.y + threadIdx.y;

    struct Synapse *row_a = (struct Synapse *) ((char*)d_synapses + neuron * pitch);
    printf("neuron: %d, synapse: %d, weight: %.2f, activity: %.2f, input: %d\n", neuron, synapse, row_a[synapse].weight, row_a[synapse].activity, row_a[synapse].input);
}

__global__ void printNeurons(int *d_neuron_outputs, float *d_weighted_sums){
    int neuron = blockIdx.x*blockDim.x + threadIdx.x;
    printf("neuron: %d, weighted sum: %.2f, output: %d\n", neuron, d_weighted_sums[neuron], d_neuron_outputs[neuron]);
}

int main(void){
    //initialize curand
    curandState_t d_curand_state;
    cudaMalloc((void**) &d_curand_state, sizeof(curandState_t));
    init_random_seed<<<1, 1>>>(time(0), d_curand_state);
    
    //allocate memory for neurons
    size_t dev_pitch;
    struct Synapse *d_synapses;
    int *d_neuron_outputs;
    float *d_weighted_sums;
    int *d_brain_inputs;
    cudaMalloc(&d_brain_inputs, sizeof(int) * NUM_INPUTS);
    cudaMalloc(&d_weighted_sums, sizeof(float) * NUM_NEURONS);
    cudaMalloc(&d_neuron_outputs, sizeof(int) * NUM_NEURONS);
    cudaMallocPitch(&d_synapses, &dev_pitch, NUM_SYNAPSES_PER_NEURON * sizeof(struct Synapse), NUM_NEURONS);
    
    dim3 synapses_dim(NUM_SYNAPSES_PER_NEURON, NUM_NEURONS, 1);

    // initialize brain
    init_synapses<<<1, synapses_dim>>>(d_synapses, dev_pitch, d_neuron_outputs, d_brain_inputs, d_curand_state);
    init_neurons<<<1, NUM_NEURONS>>>(d_neuron_outputs, d_weighted_sums);
    cudaDeviceSynchronize();
    
    //set brain inputs
    int inputs[] = {1, 0, 1, 1};
    cudaMemcpy(d_brain_inputs, inputs, sizeof(int) * NUM_INPUTS, cudaMemcpyHostToDevice);
    
    compute_synapses<<<1, synapses_dim>>>(d_synapses, d_weighted_sums, dev_pitch);
    cudaDeviceSynchronize();
    
    compute_neurons<<<1, NUM_NEURONS>>>(d_neuron_outputs, d_weighted_sums);
    cudaDeviceSynchronize();
    
    //printSynapses<<<1, synapses_dim>>>(d_synapses, dev_pitch);
    //printNeurons<<<1, NUM_NEURONS>>>(d_neuron_outputs, d_weighted_sums);
    cudaDeviceSynchronize();
    
    return 0;
}
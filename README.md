# Spiking Artificial Neural Network

An attempt to build a simple spiking artificial neural network with a reward-driven Hebbian learning function. Since it doesn't rely on backpropagation it can be wired completely randomly. Synapses with low weights will be destroyed and replaced by a random connection. This should enable the ANN to evolve and optimize its own architecture. It can, for example, form a highly recurrent network which is similar to the  biological cerebrum.

Scores in CartPole envionment with _max_episode_steps = 200
![alt text](https://github.com/AlexanderKoch-Koch/EvolvingANN/blob/LTD/ANN_Performance.PNG "Learning performance in CartPole environment")

# Python CUDA C extension
cuda/EvolvingANN.c contains the Python interface. All the functions directly accessible from Python are defined here. The first function call should always be init(). This calls the init() function in Brain.c which initializes all variables for the ANN.
The function think() in Brain.c causes one compute step for all neurons (read inputs -> compute output -> store output - > tag active synapses). The output is determined by a simple threshold function with some randomness to improve exploration. This randomness will be slowly decreased to improve exploitation near the end of training. Additionally, the connections between neurons are changed if required. Connections with a low absolute value are removed and replaced by a new random connection. The weight will be again initialized randomly.
In order to maximize reward, Brain.c contains a function called process_reward(). This changes the synapse weights according to the following formula: weight += learning_rate * recent_synapse_activity * reward. recent_synapse_activity will not be reset since it might have caused a reward which has not yet been processed. It will only be reset when reset_memory is called. This necessary for Simulations like CartPole in which the agent can "die".

# Hyperparameter optimization through an evolutionary algorithm
Evolution.py tries to find the optimal hyperparameters through random mutation. After each generation, the best agents are selected for the mating pool. The child agents will then receive random parameters from this pool. A specific percentage of these child parameters will additionally be mutated.

# Gym
This folder contains Python scripts to test the CUDA extension in the OpenAI gym. Currently there are scripts for the cartPole and for the half cheetah environment. The algorithm is able to solve the CartPole environment. However, it shows no sign of learning in the half cheetah simulation. This is probably caused by larger observation and action space.

# Example
![alt text](https://github.com/AlexanderKoch-Koch/EvolvingANN/blob/master/Example_Connectome.png "example connectome")

This is the connectome after playing 10 episodes of CartPole-v0. The circles represent the neurons. If they are green, they are currently firing. Input neurons only represent the input value. Their output is just the brain input. All the others are normal neurons. The output of the output neuron is used as an action which is fed into the environment. Otherwise, they are inactive. The thickness of the lines represents the absolute value of the related weight. Green means positive and red is negative.


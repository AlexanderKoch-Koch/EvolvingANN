# Spiking Artificial Neural Network

An attempt to build a self optimizing spiking artificial neural network with reward driven hebbian learning. Currently, it is tested in OpenAis cartPole environment. The network is implemented in C code which is compiled as a python package. This makes it easier to train the agent. The python file Simulation.py, for instance, uses the C extension to master OpenAIs CartPole-v0 environment. In the beginning, the ANN was programmed in Python for testing. Due to performance issues, However, it was reprogrammed in C. The Python prototype code is now inside Python/PythonPrototype directory.

# The C extension
C/SpikingANN.c contains the Python interface. All the functions directly accessible from Python are define here. The first function call should always be init(). This calls the init() function in Brain.c which initializes all variables for the ANN.
The function think() in Brain.c causes one compute step for all neurons (read inputs -> compute output -> store output - > tag active synapses). Additionally, the connections between neurons are change if required. Connections with a low absolute value are removed and replaced by a new random connection. The weight will be again initialized randomly.
In order to maximize reward, Brain.c contains a function called process_reward(). This changes synapse weights according to the following formula: weight += learning_rate * recent_synapse_activity * reward. recent_synapse_activity will not be reset since it might have caused a reward which has not yet been processed. It will only be reset when reset_memory is called. This necessary for Simulations like CartPole in which the agent can "die".

# Python test code
Python/Simulation.py currently provides an interface to the CartPole environment for the C extension.

# Future plans
This ANN of course has to run on a GPU to achieve some reasonable performance. NVIDIAs CUDA platform would be suitable. However this AI approach should first be tested on a smaller scale.

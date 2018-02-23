# Spiking Artificial Neural Network

An attempt to build a self optimizing spiking artificial neural network with reward driven hebbian learning. Currently, it is tested in OpenAis cartPole environment.
Simulation.py contains the interaction between the ANN and the environment. All parameters are initilized here. 
It creates a Brain object which manages all the neurons. When the brain object receives inputs from the environment, it activates the corresponding input neurons. After that, all neurons read and save their new inputs. This prohibits using wrong input values. Then the input neurons are deactivated. For a specified number of think_steps the brain repeats the computation off all neurons.
The neurons fire simply at a specific threshold.

# Weight update formula
weight += learning_rate * recent_synapse_activity * reward

The recent_synapse_activity is increased every time the synapse is activated. This variable is multiplied by a discount factor before each computation step.

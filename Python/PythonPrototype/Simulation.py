import gym
from Brain import Brain

params = {
    "num_inputs": 4,                    # number of inputs to the brain
    "num_outputs": 1,                   # number of outputs from the brain
    "TFR": 10,                          # I don't know :)
    "regeneration_rate": 0,             # probability of adding new neurons per thinking step
    "reconnection_rate": 0.3,             # probability of adding new synapse to neurons
    "learning_rate": 0.03,              # factor for weight change
    "initial_firing_threshold": 1,    # start value for neuron fire threshold
    "think_steps": 2,                  # neuron compute steps per environment step
    "firing_threshold_factor": 0.0,     # factor for firing threshold change
    "target_firing_ratio": 0.4,         # brain will optimize the neuron firing threshold to achieve this ratio
    "synapse_activity_discount": 0.9,   # discount of synapse tag per thinking step
    "initial_weight": 1.1,
    "target_weight_sum": 2
}

env = gym.make('CartPole-v0')
brain = Brain(params=params)

for i in range(100):
    steps = 0
    is_done = False
    observation = env.reset()
    while not is_done:
        activated_inputs = []
        if observation[2] > 0:
            activated_inputs.append(1)
        else:
            activated_inputs.append(0)

        if observation[3] > 0:
            activated_inputs.append(3)
        else:
            activated_inputs.append(2)

        output = brain.think(activated_inputs)
        if output is None:
            action = 0
        elif output == 0:
            action = 1

        observation, reward, is_done, info = env.step(action)
        if is_done:
            reward = -0.2
        else:
            reward = 0.01
        brain.learn(reward)
        env.render()
        steps += 1

    brain.terminate_episode()

    print(steps)

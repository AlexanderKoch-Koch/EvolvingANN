from mathai import relu, sigmoid
import numpy as np
import gym
from Neuron import Neuron
from InternalReward import InternalReward

episodes = 200

env = gym.make('CartPole-v0')
num_inputs = 4
env.render()

neuron1 = Neuron()
for i in range(num_inputs):
    neuron1.add_input()

internalReward = InternalReward(num_inputs)

for episode in range(episodes):
    is_done = False
    observation = env.reset()
    steps = 0
    while not is_done:
        env.render()
        inputs = []
        if observation[2] > 0:
            inputs.append(1)
            inputs.append(0)
        else:
            inputs.append(0)
            inputs.append(1)

        for i in range(num_inputs):
            neuron1.set_input(i, observation[i])

        result = neuron1.compute()
        action = result
        observation, reward, is_done, info = env.step(action)
        if is_done:
            reward = -0.3
        else:
            reward = 0.1

        internal_reward = internalReward.get_internal_reward(observation)
        neuron1.learn(internal_reward)
        steps += 1
        internalReward.add_experience(observation, reward)

    print(steps)
    internalReward.forget()

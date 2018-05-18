import eann
import gym
import time
from timeit import default_timer as timer


def float_to_binary_list(float_value, precision, len_list):
    result_list = []
    for i in range(int(len_list / 2)):
        if float_value < -i * precision:
            result_list.append(1)
        else:
            result_list.append(0)

        if float_value > i * precision:
            result_list.append(1)
        else:
            result_list.append(0)
    return result_list


def preprocess_inputs(observation):
    inputs = []
    inputs += float_to_binary_list(observation[0], 0.05, 10)
    inputs += float_to_binary_list(observation[1], 0.05, 4)
    inputs += float_to_binary_list(observation[2], 0.05, 8)
    inputs += float_to_binary_list(observation[3], 0.05, 4)
    return inputs

env = gym.make('CartPole-v0')

print("brain input size: " + str(len(preprocess_inputs(env.reset()))))
eann.init()

for i in range(10000):
    start_iteration = timer()
    steps = 0
    is_done = False
    observation = env.reset()
    while not is_done:
        start = timer()
        inputs = preprocess_inputs(observation)
        
        output = eann.think(inputs)
        elapsed = timer()
        elapsed = elapsed - start
        #print("compute: " + str(elapsed) + "s")
        #print(output[0])
        if output[0] > 0:
            action = 1
        else:
            action = 0
        observation, reward, is_done, info = env.step(action)
        if is_done:
            reward = -0.2
        else:
            reward = 0.005

        #env.render()
        start = timer()
        eann.reward(reward)
        
        steps += 1
        elapsed = timer()
        elapsed = elapsed - start
        #print("reward processing: " + str(elapsed) + "s")
    
    eann.reset_memory()
    elapsed = timer() - start_iteration
    print("result: " + str(steps) + "   " + str(elapsed/steps) + "s per step")

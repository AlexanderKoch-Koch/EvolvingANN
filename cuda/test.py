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
    
def binary_list_to_float(binary_list, mean, value_per_bit):
    size = len(binary_list)
    value = mean - value_per_bit * 0.5 * float(size)
    for i in binary_list:
        value += i * value_per_bit
    return value
    


def preprocess_inputs(observation):
    inputs = []
    for i in range(24):
        inputs += float_to_binary_list(observation[i], 0.05, 8)

    return inputs

env = gym.make('BipedalWalker-v2')

print("observation size: " + str(env.action_space.shape[0]))
print("brain input size: " + str(len(preprocess_inputs(env.reset()))))
eann.init()

for i in range(10000):
    start_iteration = timer()
    steps = 0
    reward_sum = 0
    is_done = False
    observation = env.reset()
    while steps < 200 and not is_done:
        start = timer()
        #print(preprocess_inputs(observation))
        inputs = preprocess_inputs(preprocess_inputs(observation))
        
        outputs = eann.think(inputs)
        elapsed = timer()
        elapsed = elapsed - start
        #print("compute: " + str(elapsed) + "s")
        
        actions = []
        actions.append(binary_list_to_float(outputs[0:9], 0, 0.2))
        actions.append(binary_list_to_float(outputs[10:19], 0, 0.2))
        actions.append(binary_list_to_float(outputs[20:29], 0, 0.2))
        actions.append(binary_list_to_float(outputs[30:39], 0, 0.2))
        observation, reward, is_done, info = env.step(actions)
        if reward == -100:
            # agent  has fallen
            reward = -2
        else:
            reward_sum += reward

        #env.render()
        start = timer()
        #print("reward: " + str(reward))
        eann.reward(reward)
        time.sleep(0.01)
        steps += 1
        elapsed = timer()
        elapsed = elapsed - start
        #print("reward processing: " + str(elapsed) + "s")
    
    eann.reset_memory()
    elapsed = timer() - start_iteration
    print("distance: " + str(reward_sum) + "\tsteps: " + str(steps) + "\t" + str(elapsed/steps) + "s per step")
    #print(reward_sum)
import spikingann
import gym
import time


def float_to_binary_list(float_value, precision, len_list):
    result_list = []
    for i in range(int(len_list/2)):
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
    inputs += float_to_binary_list(observation[0], 0.05, 4)
    inputs += float_to_binary_list(observation[1], 0.05, 4)
    inputs += float_to_binary_list(observation[2], 0.05, 8)
    inputs += float_to_binary_list(observation[3], 0.05, 4)
    return inputs


env = gym.make('CartPole-v0')
num_inputs = len(preprocess_inputs(env.reset()))
print(type(num_inputs))
parameters = {
    "num_inputs": num_inputs,
    "num_neurons": 5,
    "num_outputs": 1,
    "num_synapses_per_neuron": 13,
    "learning_rate": 0.5,
    "threshold": 0,
    "activity_discount_factor": 0.9,
    "max_weight_value": 0.5,
    "max_start_weight_sum": 2,
    "min_weight": 0.01
}
print(parameters)
spikingann.init(parameters)

for i in range(1000):
    steps = 0
    is_done = False
    observation = env.reset()
    while not is_done:
        start = time.clock()
        output = spikingann.think(preprocess_inputs(observation))
        elapsed = time.clock()
        elapsed = elapsed - start
        #print(str(elapsed) + "s")
        print("output: " + str(output))
        if output[0] > 0:
            action = 1
        else:
            action = 0

        observation, reward, is_done, info = env.step(action)
        if is_done:
            reward = -0.2
        else:
            reward = 0.005

        spikingann.reward(reward)

        env.render()
        time.sleep(0.01)
        steps += 1

    spikingann.reset_memory()

    # print("Episode " + str(i) + " steps= " + str(steps))

import spikingann
import gym
import time


def float_to_binary_list(float_value, precision, len_list):
    result_list = []
    print("float_value: " + str(float_value))
    for i in range(int(len_list/2)):
        if float_value < -i * precision:
            result_list.append(1)
        else:
            result_list.append(0)

        if float_value > i * precision:
            result_list.append(1)
        else:
            result_list.append(0)
    print("result_list: " + str(result_list))
    return result_list


def preprocess_inputs(observation):
    inputs = []
    inputs += float_to_binary_list(observation[2], 0.05, 10)
    return inputs


env = gym.make('CartPole-v0')
print(preprocess_inputs(env.reset()))
num_inputs = len(preprocess_inputs(env.reset()))
spikingann.init(10, num_inputs, 1)

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

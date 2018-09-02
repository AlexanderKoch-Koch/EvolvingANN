import sys
import eann
import gym
import time
from float_binary_conversion import float_to_binary_list, binary_list_to_float
from Preprocessing import preprocess_inputs

env = gym.make('HalfCheetah-v2')
action_len = env.action_space.shape[0]
print("brain input size: " + str(len(preprocess_inputs(env.reset()))))
assert(eann.init("./tensorboard_logs/cart_pole") == 0)

for i in range(10000):
    steps = 0
    is_done = False
    observation = env.reset()
    while not is_done:
        inputs = []
        for i in range(env.observation_space.shape[0]):
            inputs += float_to_binary_list(observation[i], 0.05, 5)

        start = time.clock()
        output = eann.think(inputs)
        elapsed = time.clock()
        elapsed = elapsed - start
        # print("\n" + str(elapsed) + "s")
        #print(output[0])
        actions = []
        output_len = len(output)
        if output_len < action_len:
            print("eann think output is too short. Please adjust the num_outputs in the Hyperparameters file")
            print("The environment takes an action vector of length {}".format(action_len))
            sys.exit(-1)
        print("eann output: ")
        print(output)
        for i in range(action_len):
            print(action_len)
            num_outputs = output_len // action_len
            actions.append(binary_list_to_float(output[i*num_outputs:(i+1) * num_outputs], -1, 1))
        
        print(actions)
        observation, reward, is_done, info = env.step(actions)

        #env.render()
        eann.reward(reward)
        time.sleep(0.01)
        steps += 1

    #eann.reset_memory()
    print("result: " + str(steps))
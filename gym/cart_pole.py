import eann
import gym
import time
from float_binary_conversion import float_to_binary_list
from Preprocessing import preprocess_inputs

env = gym.make('CartPole-v0')

print("brain input size: " + str(len(preprocess_inputs(env.reset()))))
eann.init("./tensorboard_logs/cart_pole")

for i in range(10000):
    steps = 0
    is_done = False
    observation = env.reset()
    while not is_done:
        inputs = []
        inputs += float_to_binary_list(observation[0], 0.05, 10)
        inputs += float_to_binary_list(observation[1], 0.05, 4)
        inputs += float_to_binary_list(observation[2], 0.05, 8)
        inputs += float_to_binary_list(observation[3], 0.05, 4)
        start = time.clock()
        output = eann.think(inputs)
        elapsed = time.clock()
        elapsed = elapsed - start
        # print("\n" + str(elapsed) + "s")
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
        eann.reward(reward)
        time.sleep(0.01)
        steps += 1

    #eann.reset_memory()
    print("result: " + str(steps))
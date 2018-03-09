import spikingann
import gym
import time


def observation_to_binary(observation):
    activated_inputs = []
    if observation[2] > 0:
        activated_inputs.append(1)
        activated_inputs.append(0)
        if observation[2] > 0.5:
            activated_inputs.append(1)
            activated_inputs.append(0)
    else:
        activated_inputs.append(0)
        activated_inputs.append(1)
        if observation[2] < -0.5:
            activated_inputs.append(0)
            activated_inputs.append(1)

    return activated_inputs


env = gym.make('CartPole-v0')
num_inputs = len(env.reset())
spikingann.init(2, num_inputs, 1)

for i in range(1000):
    steps = 0
    is_done = False
    observation = env.reset()
    while not is_done:
        start = time.clock()
        output = spikingann.think(observation)
        elapsed = time.clock()
        elapsed = elapsed - start
        #print(str(elapsed) + "s")
        #print(output)
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
        time.sleep(0.05)
        steps += 1

    spikingann.reset_memory()

    print("Episode " + str(i) + " steps= " + str(steps))

import spikingann
import gym

spikingann.init(10, 2, 1)

env = gym.make('CartPole-v0')

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
            activated_inputs.append(1)
        else:
            activated_inputs.append(0)

        output = spikingann.think(activated_inputs)
        print(output)
        output = output[0]
        if output is None:
            action = 0
        elif output == 0:
            action = 1

        observation, reward, is_done, info = env.step(action)
        if is_done:
            reward = -0.2
        else:
            reward = 0.01
            spikingann.reward(reward)
        env.render()
        steps += 1

    spikingann.reset_memory()

    print(steps)

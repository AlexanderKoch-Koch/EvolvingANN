import gym
from Brain import Brain


env = gym.make('CartPole-v0')

params = {
    "num_inputs": 2,
    "num_outputs": 1,
    "TFR": 10,
    "regeneration_rate": 0,
    "reconnection_rate": 0,
    "learning_rate": 0.01,
    "initial_firing_threshold": 0.5
}

brain = Brain(params=params)

for i in range(100):
    steps = 0
    is_done = False
    observation = env.reset()
    while not is_done:
        # draw connectome every 20 steps
        if steps % 20 == 0:
            brain.draw_connectome()

        if observation[2] > 0:
            output = brain.think(10, [1])
        else:
            output = brain.think(10, [0])

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

    print(steps)

import gym
from Brain import Brain


class EvolvingANN:

    def __init__(self, environment):
        self.env = environment
        self.brain = Brain(2, 1)
        self.steps = None

    def start_episode(self):
        self.steps = 0
        is_done = False
        observation = self.env.reset()
        while not is_done:
            # draw connectome every 10 steps
            if self.steps % 10 == 0:
                self.brain.draw_connectome()

            if observation[2] > 0:
                output = self.brain.think(10, [1])
            else:
                output = self.brain.think(10, [0])

            if output is None:
                action = 0
            elif output == 0:
                action = 1

            observation, reward, is_done, info = env.step(action)
            if is_done:
                reward = -0.2
            else:
                reward = 0.01
            self.brain.learn(reward)
            env.render()
            self.steps += 1
        return self.steps


env = gym.make('CartPole-v0')
evolvingANN = EvolvingANN(env)
for i in range(100):
    print("score = " + str(evolvingANN.start_episode()))

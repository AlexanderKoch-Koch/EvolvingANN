import spikingann
import gym
import time
import collections
from Preprocessing import preprocess_inputs


class Agent:

    def __init__(self, params):
        self.params = params

    def run(self, queue):
        env = gym.make('CartPole-v0')
        num_inputs = len(preprocess_inputs(env.reset()))
        brain_parameters = {
            "num_neurons": self.params[0],
            "num_outputs": 1,
            "num_synapses_per_neuron": self.params[2],
            "learning_rate": self.params[3],
            "threshold": self.params[4],
            "activity_discount_factor": self.params[5],
            "max_weight_value": self.params[6],
            "max_start_weight_sum": self.params[7],
            "min_weight": self.params[8],
            "num_inputs": num_inputs
        }

        print(brain_parameters)
        spikingann.init(brain_parameters)
        scores_deque = collections.deque(maxlen=10)

        for i in range(200):
            steps = 0
            is_done = False
            observation = env.reset()
            while not is_done:
                output = spikingann.think(preprocess_inputs(observation))
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
                steps += 1

            spikingann.reset_memory()
            scores_deque.append(steps)

        queue.put(sum(scores_deque)/len(scores_deque))

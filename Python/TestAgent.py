import spikingann
import gym
import time
import collections
from Preprocessing import preprocess_inputs


class Agent:

    def __init__(self, params):
        self.params = params

    def run(self):
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

        print(self.params)
        print(brain_parameters)
        spikingann.init(brain_parameters)
        scores_deque = collections.deque(maxlen=10)

        for i in range(1000):
            steps = 0
            is_done = False
            observation = env.reset()
            while not is_done:
                # start = time.clock()
                output = spikingann.think(preprocess_inputs(observation))
                # elapsed = time.clock()
                # elapsed = elapsed - start
                # print(str(elapsed) + "s")
                if output[0] > 0:
                    action = 1
                else:
                    action = 0

                observation, reward, is_done, info = env.step(action)
                if is_done:
                    reward = -0.2
                else:
                    reward = 0.005

                env.render()
                spikingann.reward(reward)
                time.sleep(0.01)
                steps += 1

            spikingann.reset_memory()
            scores_deque.append(steps)

        print(sum(scores_deque)/len(scores_deque))


params = [
    10.1382227e+00,
    1.0505338e+00,
    1.2871541e+01,
    4.8885125e-01,
    0.0000000e+00,
    8.9562744e-01,
    5.2541727e-01,
    2.1306219e+00,
    9.5756426e-03,
    2.0000000e+02
]
agent = Agent(params)
agent.run()


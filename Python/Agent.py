import spikingann
import gym
import time
import collections


class Agent:

    def __init__(self, params):
        self.params = params

    def run(self, queue):
        env = gym.make('CartPole-v0')
        num_inputs = len(self.preprocess_inputs(env.reset()))
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

        #print(self.params)
        spikingann.init(brain_parameters)
        scores_deque = collections.deque(maxlen=10)

        for i in range(1000):
            steps = 0
            is_done = False
            observation = env.reset()
            while not is_done:
                start = time.clock()
                output = spikingann.think(self.preprocess_inputs(observation))
                elapsed = time.clock()
                elapsed = elapsed - start
                # print(str(elapsed) + "s")
                #print("output: " + str(output))
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
                spikingann.reward(reward)
                # time.sleep(0.01)
                steps += 1

            spikingann.reset_memory()
            scores_deque.append(steps)

        queue.put(sum(scores_deque)/len(scores_deque))

    def float_to_binary_list(self, float_value, precision, len_list):
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

    def preprocess_inputs(self, observation):
        inputs = []
        inputs += self.float_to_binary_list(observation[0], 0.05, 4)
        inputs += self.float_to_binary_list(observation[1], 0.05, 4)
        inputs += self.float_to_binary_list(observation[2], 0.05, 8)
        inputs += self.float_to_binary_list(observation[3], 0.05, 4)
        return inputs


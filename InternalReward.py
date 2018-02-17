from sklearn.neural_network import MLPRegressor
from collections import deque


class InternalReward:

    def __init__(self, observation_size):
        self.observation_size = observation_size
        self.experiences = deque([], 5)
        self.dnn = MLPRegressor(hidden_layer_sizes=(observation_size, 1), random_state = 1)
        self.dnn.fit([[0] * observation_size], [0])

    def add_experience(self, observation, reward):
        experience = observation.tolist()
        experience.append(reward)
        self.experiences.append(experience)

        if len(self.experiences) >= 5:
            X = [self.experiences[0][0:self.observation_size]]
            rewards = [x[self.observation_size] for x in self.experiences]
            y = [sum(rewards)]
            print(X)
            print(y)
            self.dnn.fit(X, y)

    def get_internal_reward(self, observation):
        prediction = self.dnn.predict([observation])
        print(prediction)
        return prediction

    def forget(self):
        self.experiences.clear()

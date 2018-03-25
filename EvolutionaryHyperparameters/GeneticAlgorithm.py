import random
import numpy as np
import math


def mutate(params, sigma_divisor, mutate_percent):
    """chooses params randomly from gauss curve with mu=original param and sigma=mu/sigma_divisor"""
    for param in range(len(params)):
        if random.randint(0, 10000)/100 <= mutate_percent:
            params[param] = random.gauss(mu=params[param], sigma=params[param]/sigma_divisor)

    return params


def round_float_random(float_value):
    decimals = float_value % 1
    if random.randint(1, 99) <= decimals * 100:
        return math.ceil(float_value)
    else:
        return math.floor(float_value)


def create_mating_pool(agents, num_parents):
    """returns randomly chosen agents"""
    num_params = len(agents[0]) - 1
    score_index = len(agents[:][0]) - 1
    min_score = np.min(agents[:, score_index])
    agents[:, score_index] = agents[:, score_index] - min_score     # subtract minimum value
    agents[:, score_index] = agents[:, score_index] * agents[:, score_index]    # square
    agents[:, score_index] = agents[:, score_index] / np.sum(agents[:, score_index])
    selection = np.random.choice(len(agents), size=num_parents, replace=False, p=agents[:, score_index])
    mating_params = np.zeros((num_parents, num_params))
    for parent in range(num_parents):
        mating_params[parent] = agents[selection[parent]][:-1]

    return mating_params


def crossover(mating_params, num_children):
    """returns children params chosen from mating_params"""
    num_parents = len(mating_params)
    num_params = len(mating_params[0])
    params = np.zeros((num_children, num_params))
    for child in range(num_children):
        for param in range(num_params):
            params[child][param] = mating_params[random.randint(0, num_parents - 1)][param]

    return params

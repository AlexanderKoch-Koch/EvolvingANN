from Agent import Agent
import numpy as np
from multiprocessing.pool import ThreadPool
from multiprocessing import Process, Queue, cpu_count
from GeneticAlgorithm import mutate, create_mating_pool, crossover, round_float_random

if __name__ == "__main__":

    pool = ThreadPool(processes=1)

    # evolution parameters
    generations = 200
    sigma_divisor = 15
    mutate_percent = 2
    mating_pool_size = 64
    batch_size = int(cpu_count()) - 1
    num_agents = 100 * batch_size    # must be a multiple of batch size
    num_batches = int(num_agents/batch_size)
    print("batch size: " + str(batch_size))

    # start parameters for first gen agents
    params_start = [
        5,       # num_neurons
        1,       # num_outputs
        13,      # num_synapses_per_neuron
        0.5,       # learning_rate
        0,     # threshold
        0.9,        # activity_discount_factor
        0.5,        # max_weight_value
        2,      # max_start_weight_sum
        0.01      # min_weight
    ]

    params = np.zeros(shape=(num_agents, len(params_start)))
    # copy params_start
    params[:] = params_start
    # mutate all params
    for agent_params in params:
        agent_params = mutate(agent_params, sigma_divisor, 100)
    score_index = len(params_start)

    for generation in range(generations):
        # for each generation
        print("generation: " + str(generation))
        parameters = []
        scores = []
        queues = []
        processes = []
        agents = np.zeros((num_agents, len(params_start) + 1), dtype=np.float32)  # params, score
        current_agent = 0
        for batch in range(num_batches):
            # start processes of this batch
            for batch_agent in range(batch_size):
                i = batch_agent + batch * batch_size
                params[i] = mutate(params[i], sigma_divisor, mutate_percent)
                agents[i][0:-1] = params[i]
                agent = Agent(agents[i][0:len(agents[i] - 1)])
                queues.append(Queue())
                processes.append(Process(target=agent.run, args=(queues[i],)))
                processes[i].start()

            # wait for all processes in this batch to finish
            for batch_agent in range(batch_size):
                i = batch_agent + batch * batch_size
                processes[i].join()

        for a in range(num_agents):
            agents[a][score_index] = queues[a].get()

        # sort agents by score
        agents = agents[np.argsort(agents[:, score_index])]

        # print scores
        print(agents[:, score_index])
        print(agents[-1:, :])

        mating_params = create_mating_pool(agents, mating_pool_size)
        params = crossover(mating_params, num_agents)




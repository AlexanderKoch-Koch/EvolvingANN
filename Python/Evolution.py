from Agent import Agent
import numpy as np
from multiprocessing.pool import ThreadPool
from multiprocessing import Process, Queue, cpu_count
from GeneticAlgorithm import mutate, create_mating_pool, crossover, round_float_random

if __name__ == "__main__":

    pool = ThreadPool(processes=1)

    # evolution parameters
    generations = 50
    num_agents = 99
    sigma_divisor = 15
    mutate_percent = 5
    batch_size = int(cpu_count()) - 1
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


    #int_params_indexes = [0, 1, 2, 3, 4, 5, 6, 9, 12]

    params = np.zeros(shape=(num_agents, len(params_start)))
    # mutate params_start to each agent params
    params[:] = mutate(params_start, sigma_divisor, 95)

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
                print("current agent: " + str(i))
                params[i] = mutate(params[i], sigma_divisor, mutate_percent)
                agents[i][0:-1] = params[i]
                agent = Agent(agents[i][0:len(agents[i] - 1)])
                queues.append(Queue())
                processes.append(Process(target=agent.run, args=(queues[i],)))
                processes[i].start()

            # wait for all processes in this batch to finish
            for batch_agent in range(batch_size):
                i = batch_agent + batch * batch_size
                print("waiting for: " + str(i))
                processes[i].join()

        score_index = len(params_start)
        for a in range(num_agents):
            agents[a][score_index] = queues[a].get()

        # sort agents by score
        agents = agents[np.argsort(agents[:, score_index])]

        # print scores
        print(agents[:, score_index])
        print(agents[-1:, :])

        mating_params = create_mating_pool(agents, 16)
        params = crossover(mating_params, num_agents)




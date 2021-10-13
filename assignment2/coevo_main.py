from typing import List
import random
import numpy as np
from deap import base, creator, tools, algorithms, cma
import os, sys
import array

p = os.path.abspath('.')
sys.path.insert(1, p)

from controllers.coevo_controller import player_controller_demo
from assignment2.environment import New_Environment as Environment

def create_environment(enemies: List[int]):
    environment = Environment(
        experiment_name='test',
        multiplemode="yes",  # yes or no
        enemies=enemies,  # array with 1 to 8 items, values from 1 to 8
        level=2,  # integer
        playermode="ai",  # ai or human
        player_controller=player_controller_demo(10),
        enemymode="static",  # ai or static
        speed="fastest",  # normal or fastest
        randomini="yes",  # yes or no
    )
    return environment

def evaluate(individual, env):
    f,p,e,t = env.play(pcont=individual)
    # print('Overall results:', f,p,e,t)
    return (f,)

# --- from DEAP documentation: https://deap.readthedocs.io/en/master/examples/es_fctmin.html
def generateES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind

# --- from DEAP documentation: https://deap.readthedocs.io/en/master/examples/es_fctmin.html
def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children
        return wrappper
    return decorator

def apply_shared_fitness(population, ind):
    d_threshold = 100
    alpha = 1

    temp = 0
    for other in population:
        dist = np.linalg.norm(ind-other)
        print(dist)
        if dist < d_threshold:
            temp += 1 - (dist/d_threshold)**alpha
    shared_fitness = ind.fitness.value / temp
    return shared_fitness

def customTournamentSelection(population, k):
    # Transform all fitness values to shared fitness values
    for ind in population:
        ind.fitness.value = apply_shared_fitness(population, ind)

    # # Select individuals with tournament method
    # chosen = []
    # for i in range(len(population)):
    #     aspirants = [random.choice(population) for j in range(k)]
    #     print(aspirants)
    #     chosen.append(max([ind.fitness.value for ind in aspirants]))
    #     print(chosen)
    # return chosen

def best_individual(population):
    fitnesses = [ind.fitness.values for ind in population]
    return population[np.argmax(fitnesses)]

def run_deap(env, n_pop=20, n_gens_ga=10, n_gens_cma=5, enemies=[1,2], 
            mutpb=0.1, indpb=0.05, tournsize=3):
    """
        mutpb: The probability of mutating an individual.
        indpb: Independent probability for each attribute to be mutated.
        tournsize: Number of individuals participating in a tournament
    """
    ind_size = 265 #number of nodes in network (250 weights (20*10*5) + 15 bias (10+5))

    mu = 0.5 #Mean for the gaussian addition mutation
    sigma = 0.05 #Standard deviation for the gaussian addition mutation
    cxpb = 0.5 #The probability of mating two individuals.

    # Creation of required types
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax, strategy=None)
    creator.create("Strategy", array.array, typecode="d")

    toolbox = base.Toolbox()

    # --- partially from DEAP documentation: https://deap.readthedocs.io/en/master/examples/es_fctmin.html
    MIN_VALUE = -1
    MAX_VALUE = 1
    MIN_STRATEGY = 0.05
    MAX_STRATEGY = 1
    toolbox.register("individual", generateES, creator.Individual, creator.Strategy, ind_size, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register('mate', tools.cxOnePoint)
    # toolbox.register("mate", tools.cxESBlend, alpha=0.1)
    # toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
    toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=indpb)
    # toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    toolbox.register("evaluate", evaluate, env=env)

    # --- Tools for saving the statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Mean", np.mean)
    stats.register("Max", np.max)
    stats.register("Min", np.min)

    pop = toolbox.population(n=n_pop)
    hof = tools.HallOfFame(1, similar=np.array_equal)
    pop_ga, log_ga = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=n_gens_ga, halloffame=hof, stats=stats)

    print('\nGA is done, CMA-ES next!\n')

    # implementation of CMA-ES
    centroid = best_individual(pop_ga)
    strategy= cma.Strategy(centroid, sigma=sigma)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update) 
    pop_cma, log_cma = algorithms.eaGenerateUpdate(toolbox=toolbox, ngen=n_gens_cma, stats=stats, halloffame=hof)
    print(hof[0].fitness.values[0])
    return hof[0].fitness.values[0]

def main(n_pop=20, n_gens_ga=10, n_gens_cma=5, enemies=[1,2], mutpb=0.1, indpb=0.05, tournsize=3):
    # random.seed(20)
    # np.random.seed(20)

    print("Settings:\nmutpb", mutpb, '\nindpb', indpb, '\ntournsize:', tournsize)

    env = create_environment(enemies)
    return run_deap(env, n_pop, n_gens_ga, n_gens_cma, enemies)

if __name__ == '__main__':
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    main()
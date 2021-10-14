from typing import List
import random
import numpy as np
from deap import base, creator, tools, algorithms, cma
import os, sys
import array
import pickle

p = os.path.abspath('.')
sys.path.insert(1, p)

from controllers.deap_controller import player_controller_demo
from assignment2.environment import New_Environment as Environment

def create_environment(name: str, enemies: List[int]):
    environment = Environment(
        experiment_name=name,
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
    shared_fitness = ind.fitness.values / temp
    return shared_fitness

# def customTournamentSelection(population, k, tournsize=3):
#     # Transform all fitness values to shared fitness values
#     for ind in population:
#         ind.fitness.values = apply_shared_fitness(population, ind)

#     # # Select individuals with tournament method
#     # chosen = []
#     # for i in range(k):
#     #     aspirants = [random.choice(population) for j in range(tournsize)]
#     #     chosen.append(max([ind.fitness.values for ind in aspirants]))
#     # return chosen

def best_individual(population):
    fitnesses = [ind.fitness.values for ind in population]
    return population[np.argmax(fitnesses)]

def run_deap(env, **config):
    n_pop = config.get('n_pop', 50)
    n_gens_ga = config.get('n_gens_ga', 30)
    n_gens_cma = config.get('n_gens_cma', 15)
    mutpb = config.get('mutpb', 0.25)
    indpb = config.get('indpb', 0.25)
    tournsize = config.get('tournsize', 3)
    cxpb = config.get('cxpb', 0.5)
    sigma = config.get('sigma', 0.2)
    mu = config.get('mu', 0.0)
    optimize = config.get('optimize', False)
    """
        n_pop: The population size
        n_gens_ga: The number of generations for ga
        n_gens_cma: The number of generations for cma
        mutpb: The probability of mutating an individual.
        indpb: Independent probability for each attribute to be mutated.
        tournsize: Number of individuals participating in a tournament
        cxpb: The probability of mating two individuals.
        mu: The mean for Gaussian operations
        sigma: The standard deviation for Gaussian operations.
        optimize: A boolean value for running the optimize algorithm
    """

    print('Setting up DEAP!\n')

    ind_size = 265 #number of nodes in network (250 weights (20*10*5) + 15 bias (10+5))

    # Creation of required types
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax, strategy=None)
    creator.create("Strategy", array.array, typecode="d")

    toolbox = base.Toolbox()

    # --- partially from DEAP documentation: https://deap.readthedocs.io/en/master/examples/es_fctmin.html
    MIN_VALUE = -1
    MAX_VALUE = 1
    MIN_STRATEGY = 0.01
    MAX_STRATEGY = 0.5
    toolbox.register("individual", generateES, creator.Individual, creator.Strategy, ind_size, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register('mate', tools.cxOnePoint)
    # toolbox.register("mate", tools.cxESBlend, alpha=0.1)
    # toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
    toolbox.register("mutate", tools.mutGaussian, mu=mu, sigma=sigma, indpb=indpb)
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

    print('Setup is complete, now starting GA!\n')

    pop_ga, log_ga = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=n_gens_ga, halloffame=hof, stats=stats)

    print('GA is done, CMA-ES next!')

    # implementation of CMA-ES
    centroid = best_individual(pop_ga)
    strategy= cma.Strategy(centroid, sigma=sigma)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update) 
    pop_cma, log_cma = algorithms.eaGenerateUpdate(toolbox=toolbox, ngen=n_gens_cma, stats=stats, halloffame=hof)

    stats_per_gen = []

    for gen in log_ga:
        stats_per_gen.append([gen['Mean'], gen['Max']])

    for gen in log_cma:
        stats_per_gen.append([gen['Mean'], gen['Max']])

    if optimize:
        return hof[0].fitness.values[0]

    return stats_per_gen, hof[0]

def main(**config):
    experiment_name = config.get('name', 'test')
    enemies = config.get('enemies', [5,8])
    random.seed(10)
    np.random.seed(10)
    env = create_environment(experiment_name, enemies)
    return run_deap(env, **config)

if __name__ == '__main__':
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    config = dict()
    main(**config)
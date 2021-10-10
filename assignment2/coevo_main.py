import numpy as np
import random

from typing import List
from assignment2.controllers.co_evo_controller import player_controller
from assignment1.environment import New_Environment as Environment
from deap import base, creator, tools, algorithms

def create_environment(enemies: List[int]):
    environment = Environment(
        experiment_name='test',
        multiplemode="no",  # yes or no
        enemies=enemies,  # array with 1 to 8 items, values from 1 to 8
        level=2,  # integer
        playermode="ai",  # ai or human
        player_controller=player_controller,
        enemymode="static",  # ai or static
        speed="fastest",  # normal or fastest
        randomini="yes",  # yes or no
    )
    return environment

def create_DEAP(self):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("weight_bin", random.random)  # Initiate random weights
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.weight_bin, n=265)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.01)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", self.evaluate)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Mean", np.mean)
    stats.register("Max", np.max)
    stats.register("Min", np.min)

    pop = toolbox.population(n=self.population)
    hof = tools.HallOfFame(1)

    return pop

def evaluate_individual(env, individual):
    f,p,e,t = env.play(np.asarray(individual))
    return f


def run_deap():
    env = create_environment([2])
    create_DEAP()
    deap.algorithms


if __name__ == '__main__':
    env = create_environment(2)





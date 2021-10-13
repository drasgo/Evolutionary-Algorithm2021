import os
import math 

from numpy.ma import MaskedArray
import sklearn.utils.fixes
sklearn.utils.fixes.MaskedArray = MaskedArray

from skopt import gp_minimize
from coevo_main import main

def optimize(runs: int = 2):
    mutpb = (0.05, 0.5) #The probability of mutating an individual.
    indpb = (0.05, 0.8) #The probability of mating two individuals.
    tournsize = (2, 4) #Number of individuals considered during selection
    print(f"Result {gp_minimize(single_run, [mutpb, indpb, tournsize])}")

def single_run(inputs) -> float:
    n_gens_ga = 5 #Number of generations for the ga algorithm
    n_gens_cma = math.ceil(n_gens_ga/2) #Number of generations for the cma algorithm
    n_pop = 10 #Number of individuals in the population
    enemies = [5, 8] #Subset of bosses
    return -main(n_pop, n_gens_ga, n_gens_cma, enemies, inputs[0], inputs[1], inputs[2])

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

if __name__ == '__main__':
    optimize()
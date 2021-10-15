import os
import math 
import csv
import json

from numpy.ma import MaskedArray
import sklearn.utils.fixes
sklearn.utils.fixes.MaskedArray = MaskedArray

from skopt import gp_minimize
from skopt import dump
from deap_main import main

config = {
    'name': 'optimization',
    'optimize': True,
    "n_pop": 10,
    "n_gens_ga": 2,
    "n_gens_cma": 1, 
    'enemies': [1,2],
    'tournsize': 3
}

def optimize():
    save_path = 'assignment2/experiments/deap/results'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    mutpb = (0.05, 0.5) #The probability of mutating an individual.
    indpb = (0.05, 0.8) #Independent probability for each attribute to be mutated.
    cxpb = (0.2, 0.9) #The probability of mating two individuals.
    sigma = (0.01, 0.6) #The standard deviation for Gaussian operations
    start_values = [0.2, 0.2, 0.5,0.2]
    result = gp_minimize(single_run, [mutpb, indpb, cxpb, sigma], n_calls=11, x0=start_values, random_state=10)
    print(f"Result {result}")
    dump(result, save_path + '/optimization_result.plk')
    config['mutpb'] = result.x[0]
    config['indpb'] = result.x[1]
    config['cxpb'] = result.x[2]
    config['sigma'] = result.x[3]
    config['optimize'] = False
    config['obtained_fitness'] = result.fun

    with open(save_path + '/optimization_result.json', 'w') as fp:
        json.dump(config, fp)

def single_run(inputs) -> float:
    config['mutpb'] = inputs[0]
    config['indpb'] = inputs[1]
    config['cxpb'] = inputs[2]
    config['sigma'] = inputs[3]
    return -main(**config)

if __name__ == '__main__':
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    optimize()
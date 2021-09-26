import numpy as np
import pygad.gann as gann
import pygad.nn as nn

from pygad import GA

from evoman.controller import Controller

fitnesses = {}

def fitness_func(solution, sol_idx):
    return fitnesses[sol_idx]

class ga_controller(Controller):
    def __init__(self, population: int, generations: int):
        self.networks = gann.GANN(
            num_solutions = population,
            num_neurons_input = 20,
            num_neurons_output = 5,
            output_activation = 'sigmoid')
        
        self.networks.create_population()
        self.initial_population_vector = gann.population_as_vectors(self.networks.population_networks)
        self.current_solution = 0

        self.algorithm = GA(
            num_generations = generations,
            num_parents_mating = 2,
            initial_population = self.initial_population_vector.copy(),
            fitness_func = fitness_func,
            parent_selection_type = "tournament",
            keep_parents = 2,
            K_tournament = 2,
            crossover_type = "single_point",
            crossover_probability = 0.2,
            mutation_probability = 0.05,
            mutation_percent_genes = 0.01,
            allow_duplicate_genes = False)

    def execute(self, solution_index, fitness):
        fitnesses[solution_index] = fitness
        self.current_solution += 1
        self.algorithm.run()

    def control(self, inputs: np.ndarray, controller=None):
        prediction = nn.predict(last_layer = self.networks.population_networks[self.current_solution], data_inputs = inputs)
        print(prediction)
        action = [1 if value > 0.5 else 0 for value in prediction]
        print(action)
        return action

    def evolve_networks(self):
        print(fitnesses)
        fitnesses.clear()
        self.current_solution = 0
        return None
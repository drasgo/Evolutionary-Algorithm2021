import pygad.gann as gann
from pygad import GA
from pygad.nn import layers_weights

from demo_controller import player_controller

fitnesses = {}

class ga_controller(player_controller):
    def __init__(self, population: int, generations: int):
        super().__init__(0)

        self.networks = gann.GANN(
            num_solutions = population,
            num_neurons_input = 20,
            num_neurons_output = 5,
            output_activation = 'sigmoid')
        
        self.networks.create_population()
        self.initial_population_vector = gann.population_as_vectors(self.networks.population_networks)

        self.algorithm = GA(
            num_generations = generations,
            num_parents_mating = 2,
            initial_population = self.initial_population_vector.copy(),
            fitness_func = self.fitness_func,
            parent_selection_type = "tournament",
            keep_parents = 2,
            K_tournament = 2,
            crossover_type = "single_point",
            crossover_probability = 0.2,
            mutation_probability = 0.05,
            mutation_percent_genes = 0.01,
            allow_duplicate_genes = False)
        
    def fitness_func(solution, sol_idx):
        return fitnesses[sol_idx]

    def execute(self, solution_index, fitness):
        fitnesses.append({solution_index:fitness})
        self.algorithm.run()
        self.algorithm.plot_fitness()

    def control(self, inputs, controller=None):
        return super().control(inputs, controller=controller)

    def evolve_networks(self):
        fitnesses.clear()
        return None
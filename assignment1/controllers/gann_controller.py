from numpy.random.mtrand import f
from pygad.gann.gann import population_as_matrices
from demo_controller import player_controller
import numpy as np
import pygad.gann as gann
import pygad.nn as nn

from pygad import GA

from evoman.controller import Controller
from assignment1.environment import New_Environment as Environment

def fitness_func(solution, sol_idx):
    fitnesses = 0
    controller.current_solution =sol_idx
    for enemy in controller.enemies:
        result = controller.environment.run_single(enemy, controller, "None")
        fitnesses += result[0]
        #print(f"Fitness of solution {sol_idx} for enemy {enemy}: {result[0]}")

    total_fitness = fitnesses / len(controller.enemies)
    #if len(controller.enemies) > 1:
    print(f"Total fitness of solution {sol_idx}: {total_fitness}")
        
    return total_fitness

def callback_generation(ga_instance):
    controller.current_generation += 1

    print(f"Generation {controller.current_generation}:")

    population_matrice = gann.population_as_matrices(controller.networks.population_networks, ga_instance.population)
    controller.networks.update_population_trained_weights(population_matrice)


class ga_controller(Controller):
    def __init__(self, population: int, generations: int, enemies: list):
        self.networks = gann.GANN(
            num_solutions = population,
            num_neurons_input = 20,
            num_neurons_output = 5,
            output_activation = 'sigmoid')

        self.networks.create_population()
        self.initial_population_vector = gann.population_as_vectors(self.networks.population_networks)
        self.enemies = enemies
        self.current_generation = 0

        self.environment = Environment(
            experiment_name="ga_specialist",
            enemies=enemies,
            playermode="ai",
            player_controller=self,
            enemymode="static",
            speed="fastest")

        self.algorithm = GA(
            num_generations = generations,
            num_parents_mating = 5,
            initial_population = self.initial_population_vector.copy(),
            fitness_func = fitness_func,
            on_generation = callback_generation,
            parent_selection_type = "tournament",
            keep_parents = 5,
            K_tournament = population,
            crossover_type = "single_point",
            crossover_probability = 0.2,
            mutation_probability = 0.02,
            mutation_percent_genes = 0.01,
            allow_duplicate_genes = False)

        global controller
        controller = self

    def control(self, inputs: np.ndarray, controller=None):
        # nn.predict() returns an array with an integer for each input which represents the activated output neuron
        prediction = nn.predict(self.networks.population_networks[self.current_solution], np.array([inputs], dtype = np.float128))
        action = [0, 0, 0, 0, 0]
        action[round(prediction[0])] = 1
        return action

    def execute(self):
        self.algorithm.run()
        self.algorithm.plot_fitness()
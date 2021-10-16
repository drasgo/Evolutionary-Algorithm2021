import numpy as np
import pygad.gann as gann
import pygad.nn as nn
import os
import csv

from pygad import GA
from datetime import datetime

from evoman.controller import Controller
from assignment2.environment import New_Environment as Environment


def fitness_func(solution, sol_idx):
    fitnesses = []
    gains = []
    controller.current_solution = sol_idx

    for enemy in controller.enemies:
        fitness, playerlife, enemylife, time = controller.environment.run_single(enemy, controller, "None")
        fitnesses.append(fitness)
        gains.append(playerlife - enemylife)

    total_fitness = sum(fitnesses) / len(controller.enemies)
    controller.fitnesses.append(total_fitness)

    if total_fitness > controller.current_best_solution[0]:
        gains_of_best = []
        for enemy in [1, 2, 3, 4, 5, 6, 7, 8]:
            for idx in range(5):
                gains.clear()
                fitness, playerlife, enemylife, time = controller.environment_final.run_single(enemy, controller, "None")
                gains.append(playerlife - enemylife)
            gains_of_best.append(sum(gains) / 5)
        controller.current_best_solution = [total_fitness, gains_of_best, solution]
        
    return total_fitness


def callback_generation(ga_instance):
    controller.plotting_fitnesses.append([sum(controller.fitnesses) / len(controller.fitnesses), max(controller.fitnesses)])
    controller.fitnesses.clear()
    population_matrice = gann.population_as_matrices(controller.networks.population_networks, ga_instance.population)
    controller.networks.update_population_trained_weights(population_matrice)


class ga_controller(Controller):
    def __init__(self, population: int, generations: int, enemies: list, iteration: int, mutation_rate: float, mutation_amount: float, crossover_rate: float):
        self.networks = gann.GANN(
            num_solutions = population,
            num_neurons_input = 20,
            num_neurons_hidden_layers = [10],
            num_neurons_output = 5,
            output_activation = "sigmoid",
            hidden_activations = "sigmoid")

        self.networks.create_population()
        self.initial_population_vector = gann.population_as_vectors(self.networks.population_networks)
        self.enemies = enemies
        self.fitnesses = []
        self.current_best_solution = [0, [], []]
        self.plotting_fitnesses = []
        self.iteration = iteration

        self.environment = Environment(
            experiment_name = "ga_specialist",
            enemies = enemies,
            playermode = "ai",
            player_controller = self,
            enemymode = "static",
            speed = "fastest",
            randomini = "no",
            multiplemode = "yes")

        self.environment_final = Environment(
            experiment_name = "ga_specialist_final",
            enemies = [1, 2, 3, 4, 5, 6, 7, 8],
            playermode = "ai",
            player_controller = self,
            enemymode = "static",
            speed = "fastest",
            randomini = "no",
            multiplemode = "yes")

        self.algorithm = GA(
            num_generations = generations,
            num_parents_mating = 2,
            initial_population = self.initial_population_vector.copy(),
            fitness_func = fitness_func,
            on_generation = callback_generation,
            parent_selection_type = "tournament",
            keep_parents = 2,
            K_tournament = population,
            crossover_type = "single_point",
            crossover_probability = crossover_rate,
            mutation_probability = mutation_rate,
            mutation_percent_genes = mutation_amount,
            allow_duplicate_genes = False)

        global controller
        controller = self

    def control(self, inputs: np.ndarray, controller=None):
        # nn.predict() returns an array with an integer for each input which represents the activated output neuron
        prediction = nn.predict(self.networks.population_networks[self.current_solution], np.array([inputs]))
        action = [0, 0, 0, 0, 0]
        action[round(prediction[0])] = 1
        return action

    def execute(self):
        self.algorithm.run()

        parent_dir = os.path.dirname(os.path.dirname(__file__))
        if not os.path.exists(f"{parent_dir}/results/"):
            os.mkdir(f"{parent_dir}/results/")
        target_dir = f"{parent_dir}/results/ga"
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        
        enemy_string = f""
        for enemy in self.enemies:
            enemy_string += f"{enemy}_"
        
        with open(f"{target_dir}/ga_solution_{enemy_string}{self.iteration}_lpv.csv", "w") as lpv_file:
            writer_l = csv.writer(lpv_file)
            writer_l.writerows(self.plotting_fitnesses)

        with open(f"{target_dir}/ga_solution_{enemy_string}{self.iteration}_bpv.csv", "w") as bpv_file:
            writer_b = csv.writer(bpv_file)
            writer_b.writerows(self.current_best_solution)

        print(self.current_best_solution)
        return 100 - (self.plotting_fitnesses[len(self.plotting_fitnesses)-1] - self.plotting_fitnesses[0])

from typing import List
from numpy.random.mtrand import f
from pygad.gann.gann import population_as_matrices
from demo_controller import player_controller

import numpy as np
import pygad.gann as gann
import pygad.nn as nn
import os
import csv

from pygad import GA
from datetime import datetime

from evoman.controller import Controller
from assignment1.environment import New_Environment as Environment


def fitness_func(solution, sol_idx):
    fitnesses = 0
    controller.current_solution = sol_idx

    for enemy in controller.enemies:
        result = controller.environment.run_single(enemy, controller, "None")
        fitnesses += result[0]

    total_fitness = fitnesses / len(controller.enemies)
    controller.fitnesses.append(total_fitness)
    if total_fitness > controller.current_best[0]:
        controller.current_best = [total_fitness, solution]
        
    return total_fitness


def callback_generation(ga_instance):
    controller.current_generation += 1
    controller.plotting_fitnesses.append([sum(controller.fitnesses) / len(controller.fitnesses), max(controller.fitnesses)])
    controller.fitnesses.clear()
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
        self.fitnesses = []
        self.current_best = [0, []]
        self.plotting_fitnesses = []

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
            allow_duplicate_genes = False,
            save_best_solutions = True)

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

        timestamp = datetime.now().strftime("%y%m%d%H%M%S")
        
        with open(f"{target_dir}/ga_solution_{enemy_string}{timestamp}_lpv.csv", "w") as lpv_file:
            writer_l = csv.writer(lpv_file)
            writer_l.writerows(self.plotting_fitnesses)

        fitnesses_of_best = self.test_best_solution(self.current_best[1])

        with open(f"{target_dir}/ga_solution_{enemy_string}{timestamp}_bpv.csv", "w") as bpv_file:
            writer_b = csv.writer(bpv_file)
            writer_b.writerow(fitnesses_of_best)

    def test_best_solution(self, solution) -> List[float]:
        gains = []

        test_input_layer = nn.InputLayer(20)
        test_output_layer = nn.DenseLayer(5, test_input_layer)
        test_output_layer.initial_weights = solution
        self.networks.population_networks = [test_output_layer]
        self.current_solution = 0

        for idx in range(5):
            gains.append(self.gain_func())

        return gains

    def gain_func(self):
        gains = 0

        for enemy in self.enemies:
            result = self.environment.run_single(enemy, self, "None")
            gains += (result[1] - result[2])
            
        return (gains / len(self.enemies))
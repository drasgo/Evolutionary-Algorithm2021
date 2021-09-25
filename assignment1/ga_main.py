import os
import pygad

from typing import Callable, List, Tuple
from numpy import datetime64

from assignment1.controllers.ga_controller import ga_controller as Controller
from assignment1.environment import Environment


experiment_name = "ga_specialist"

def init_algorithm(generations: int, fitness_function: Callable, mutation_probability: int, population_size: int, number_of_parents = 2, tournament_participants = 2, kept_parents = -1, initial_population = None, parent_selection_type = "sss"):
    GA = pygad.GA(
        num_generations = generations,
        sol_per_pop = population_size,
        num_genes = 2,
        num_parents_mating = number_of_parents,
        keep_parents = kept_parents,
        fitness_func = fitness_function,
        initial_population = initial_population,
        parent_selection_type = parent_selection_type,
        K_tournament = tournament_participants,
        mutation_probability = mutation_probability,
        save_best_solutions = True,
        allow_duplicate_genes = False)
        
    GA.initialize_population(0, 4, 150, [], True)
    print(GA.population)

    return GA

def evolution_step(controller: Controller, environment: Environment) -> Tuple[float, float, float]:
    print("Starting next generation")

def execute(enemies: List[int], generations: int):
    for enemy in enemies:
        controller = init_algorithm(
            generations = 50,
            population_size = 150,
            mutation_probability = 0.1,
            fitness_function = Environment.fitness_single,
            parent_selection_type = "tournament")

        environment = Environment(
            experiment_name=f"{experiment_name}_{str(enemy)}_{datetime64.now()}",
            enemies=[enemy],
            playermode="ai",
            player_controller=controller,
            enemymode="static",
            speed="fastest")
                    
        print(f"Controller and environment for enemy {enemy} created.")

        total_fitness = []
        fitness = 0

        for generation in range(generations):
            fitness = evolution_step(controller, environment)[2]
            total_fitness.append(fitness)
            print(f"Generation {generation} had a fitness of {fitness}")

        print(f"Mean fitness vs enemy {enemy} was {sum(total_fitness)/len(total_fitness)}")
        print(f"Last generation was {fitness}")

    def get_fitness(population, index) -> float:
        Environment.fitness_single

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

Controller(150)

#execute([2], 10)

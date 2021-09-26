import os

from typing import Callable, List, Tuple

from assignment1.controllers.gann_controller import ga_controller as Controller
from assignment1.environment import New_Environment as Environment


experiment_name = "ga_specialist"

def run(enemies: List[int], generations: int):
    fitness = {}
    for enemy in enemies:
        controller = Controller(50, 20)
        fitness[enemy] = []

        for generation in range(generations):
            fitnesses = cycle(controller, enemy)
            fitness[enemy].append({generation:fitnesses})
            print(generation)
            controller.evolve_networks()

    print(fitness)

def cycle(controller: Controller, enemy: int) -> List[float]:
    fitnesses = []
    for idx, network in enumerate(controller.networks.population_networks):
        environment = Environment(
            experiment_name=f"{experiment_name}_{str(enemy)}",
            enemies=[enemy],
            playermode="ai",
            player_controller=controller,
            enemymode="static",
            speed="fastest")
            
        current_fitness, player_life, enemy_life, time = environment.run_single(enemy, controller, "None")
        
        # placeholder until fitness is normalized
        current_fitness = (100 + player_life - enemy_life) / 2

        fitnesses.append(current_fitness)
        controller.execute(idx, current_fitness)

    return fitnesses

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

run([2], 30)


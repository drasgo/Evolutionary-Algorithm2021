import os

from typing import List

from assignment1.controllers.neat_controller import Neat_Controller
from assignment1.environment import Environment


def cycle(controller: Neat_Controller, environment: Environment):
    fitness, player_life, enemy_life, time = environment.play(pcont=controller)
    controller.evolve(fitness_value=fitness)


def run(name: str, enemies_numbers: List[int], gens: int, config: str):
    for enemy in enemies_numbers:
        controller = Neat_Controller(config)
        environment = Environment(experiment_name=name + str(enemy),
                          enemies=[enemy],
                          playermode="ai",
                          player_controller=controller,
                          enemymode="static",
                          level=2,
                          speed="fastest")

        for gen in range(gens):
            cycle(controller, environment)


if __name__ == '__main__':
    # True for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    enemies = []
    generations = 150
    experiment_name = "neat_specialist"

    # Neat config file location
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "configs", "config-neat")

    run(experiment_name, enemies, generations, config_path)


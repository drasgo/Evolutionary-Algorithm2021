import os

from typing import List, Tuple

from assignment1.controllers.neat_controller import Neat_Controller
from assignment1.environment import Environment
from assignment1.plotting import plot


def cycle(controller: Neat_Controller, environment: Environment) -> List[float]:
    print("starting cycle")
    print("preparing new networks")

    controller.prepare_new_networks()
    fitness_nets, player_life_nets, enemy_life_nets, time_nets = [], [], [], []
    print("starting to play")

    for idx, net in enumerate(controller.nets):
        print(f"Game nr.{idx}")
        fitness, player_life, enemy_life, time = environment.play(pcont=net)
        controller.nets[idx].append(fitness)
        fitness_nets.append(fitness)
        player_life_nets.append(player_life)
        enemy_life_nets.append(enemy_life)
        time_nets.append(time)

    print("Finished gaming.")
    controller.evolve(fitness_value=fitness_nets)
    print("Evolution finished.")
    return fitness_nets


def run():
    # True for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    enemies_numbers = [1]
    gens = 150
    name = "neat_specialist"

    # Neat config file location
    local_dir = os.path.dirname(__file__)
    config = os.path.join(local_dir, "configs", "config-neat")
    for enemy in enemies_numbers:
        controller = Neat_Controller(config)
        environment = Environment(experiment_name=name + str(enemy),
                          enemies=[enemy],
                          playermode="ai",
                          player_controller=controller,
                          enemymode="static",
                          level=2,
                          speed="fastest")

        total_fitness = []
        fitness = 0
        for gen in range(gens):
            print(f"Gen n°.{gen} of {gens}")
            fitness = cycle(controller, environment)
            total_fitness.append(fitness)
            print(f"Finished gen n°.{gen} with fitness {fitness}")

        print(f"Against enemy {enemy} the mean fitness was {sum(total_fitness)/(len(total_fitness))}")
        plot(total_fitness)
        input()


if __name__ == '__main__':
    run()


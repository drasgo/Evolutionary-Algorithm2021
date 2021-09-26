import os

from typing import List

import neat

from assignment1.controllers.neat_controller import Neat_Controller
from assignment1.environment import New_Environment
from assignment1.plotting import line_plot, box_plot

# For Pygame: True for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Neat config file location
local_dir = os.path.dirname(__file__)
config = os.path.join(local_dir, "configs", "config-neat")

name = "neat_specialist"
enemies_numbers = [1, 2, 3, 4, 5, 6, 7, 8]
gens = 20
number_of_different_runs = 10
test = 5


def cycle(controller: Neat_Controller, environment: New_Environment) -> List[float]:
    print("starting cycle")
    print("preparing new networks")

    controller.prepare_new_networks()
    fitness_nets = []
    print("starting to play")

    for idx, net in enumerate(controller.nets):
        print(f"Game nr.{idx}")
        fitness, _, _, _ = environment.play(pcont=net)
        controller.nets[idx].append(fitness)
        fitness_nets.append(fitness)

    print("Finished gaming.")
    controller.evolve()
    print("Evolution finished.")
    return fitness_nets


def neat_test(controller, environment: New_Environment) -> float:
    """
    Test the best network of the population, with the current environment
    :param controller:
    :param environment:
    :return:
    """
    best_genome = controller.stats.best_unique_genomes(1)[0]
    best_network = neat.nn.FeedForwardNetwork.create(best_genome, controller.configs)
    means = []
    for idx in range(test):
        fitness, player_life, enemy_life, _ = environment.play(pcont=[None, None, best_network])

        means.append(player_life - enemy_life)
    return sum(means)/len(means)


def run():

    for enemy in enemies_numbers:
        print(f"Starting enemy nr°.{enemy}")
        mean_max_fitness_values = []
        mean_best_networks = []

        for run_idx in range(number_of_different_runs):
            run_mean_max_fitness = []

            controller = Neat_Controller(config)
            environment = New_Environment(experiment_name=name + str(enemy),
                                          enemies=[enemy],
                                          playermode="ai",
                                          player_controller=controller,
                                          enemymode="static",
                                          speed="fastest")

            for gen in range(gens):
                print(f"Gen n°.{gen} of {gens-1}")
                fitness = cycle(controller, environment)
                mean_fitness = sum(fitness)/len(fitness)
                max_fitness = max(fitness)
                run_mean_max_fitness.append([mean_fitness, max_fitness])
                print(f"Finished gen n°.{gen} with average fitness {mean_fitness}")

            print("Testing best network")
            mean_best_networks.append(neat_test(controller, environment))
            mean_max_fitness_values.append(run_mean_max_fitness)
            print(f"\nFinished run nr°.{run_idx} of {number_of_different_runs-1}\n")

        box_plot(f"{name}_enemy#{enemy}", mean_best_networks)
        line_plot(f"{name}_enemy#{enemy}", mean_max_fitness_values)
        print(f"Finished enemy {enemy}")


if __name__ == '__main__':
    run()


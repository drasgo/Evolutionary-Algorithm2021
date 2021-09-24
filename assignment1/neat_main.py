import os

from typing import List

import neat

from assignment1.controllers.neat_controller import Neat_Controller
from assignment1.environment import New_Environment
from assignment1.plotting import line_plot, box_plot

# True for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

enemies_numbers = [1]
gens = 150
name = "neat_specialist"
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


def neat_test(controller, environment) -> float:
    """
    Test the best network of the population, with the current environment
    :param controller:
    :param environment:
    :return:
    """
    best_genome = controller.stats.best_unique_genomes(1)
    best_network = neat.nn.FeedForwardNetwork.create(best_genome, controller.configs)
    means = []
    for idx in range(test):
        fitness, _, _, _ = environment.play(pcont=best_network)
        means.append(fitness)
    return sum(means)/len(means)


def run():
    # Neat config file location
    local_dir = os.path.dirname(__file__)
    config = os.path.join(local_dir, "configs", "config-neat")
    for enemy in enemies_numbers:
        mean_max_fitness_values = []
        mean_best_networks = []

        for run_idx in range(10):
            run_mean_max_fitness = []
            total_fitness = []

            controller = Neat_Controller(config)
            environment = New_Environment(experiment_name=name + str(enemy),
                                          enemies=[enemy],
                                          playermode="ai",
                                          player_controller=controller,
                                          enemymode="static",
                                          speed="fastest")

            for gen in range(gens):
                print(f"Gen n°.{gen} of {gens}")
                fitness = cycle(controller, environment)
                mean_fitness = sum(fitness)/len(fitness)
                max_fitness = max(fitness)
                run_mean_max_fitness.append([mean_fitness, max_fitness])
                print(f"Finished gen n°.{gen} with average fitness {mean_fitness}")


            print(f"Against enemy {enemy} the mean fitness was {sum(total_fitness)/(len(total_fitness))}")
            mean_best_networks.append(neat_test(controller, environment))
            print("Best network tested")

        line_plot(f"{name}_enemy#{enemy}", mean_max_fitness_values)
        box_plot(f"{name}_enemy#{enemy}", mean_best_networks)
        input()


if __name__ == '__main__':
    run()


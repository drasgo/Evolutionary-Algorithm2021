import os

from typing import List, Tuple

from assignment1.controllers.neat_controller import Neat_Controller
from assignment1.environment import Environment


def cycle(controller: Neat_Controller, environment: Environment) -> Tuple[float, float, float]:
    print("starting cycle")
    print("preparing new networks")
    input()
    controller.prepare_new_networks()
    fitness_nets, player_life_nets, enemy_life_nets, time_nets = [], [], [], []
    print("starting to play")
    input()
    for idx, net in enumerate(controller.nets):
        print(f"Game nr.{idx}")
        input()
        fitness, player_life, enemy_life, time = environment.play(pcont=net)
        fitness_nets.append(fitness)
        player_life_nets.append(player_life)
        enemy_life_nets.append(enemy_life)
        time_nets.append(time)

    print("Finished gaming. Evolving now.")
    input()
    controller.evolve(fitness_value=fitness_nets)
    print("Evolution finished.")
    input()
    return max(fitness_nets), min(fitness_nets), sum(fitness_nets)/len(fitness_nets)


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
        print("controller and env created")
        input()
        total_fitness = []
        fitness = 0
        for gen in range(gens):
            print(f"Gen n°.{gen}")
            input()
            max_fitness, min_fitness, fitness = cycle(controller, environment)
            total_fitness.append(fitness)
            print(f"Finished gen n°.{gen} with fitness {fitness}")

            input()
        print(f"Against enemy {enemy} the mean fitness was {sum(total_fitness)/(len(total_fitness))} with last "
              f"generation fitness being {fitness}")

        input()


if __name__ == '__main__':
    # True for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    enemies = [1]
    generations = 150
    experiment_name = "neat_specialist"

    # Neat config file location
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "configs", "config-neat")
    print("pre run")
    input()
    run(experiment_name, enemies, generations, config_path)


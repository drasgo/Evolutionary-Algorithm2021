import os
import numpy as np
import json

from plotting import line_plot
from deap_main import main

from environment import New_Environment as Environment
from controllers.deap_controller import player_controller_demo

def evolve_agents(n=2):
    base_path = 'assignment2/experiments/deap/'

    # Open the environment configuration
    config_file = base_path + 'optimization_result.json'
    with open(config_file) as f:
        config = json.load(f)

    # Check folder for controllers
    controller_path = base_path + 'controllers/'
    if not os.path.exists(controller_path):
        os.makedirs(controller_path)

    # Check folder for images
    image_path = 'experiments/deap/images/'
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    # agent_sets = [[5,8], [1,2]]
    agent_sets = [[5,8]]
    for i, enemies in enumerate(agent_sets):
        config['enemies'] = enemies

        # Check folder for controllers
        save_path = controller_path + 'group_' + str(i) + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        stats_per_run = []
        for j in range(n):
            experiment_name = 'run_' + str(j)
            config['name'] = experiment_name
            stats, best_ind = main(**config)
            stats_per_run.append(stats)
            np.savetxt(save_path + 'best_controller_' + experiment_name + '.txt', best_ind)
        
        line_plot(config['name'], stats_per_run, image_path)

def run_single_enemy(enemy, controller):
    environment = Environment(
        experiment_name='test',
        multiplemode="no",  # yes or no
        enemies=[enemy],  # array with 1 to 8 items, values from 1 to 8
        level=2,  # integer
        playermode="ai",  # ai or human
        player_controller=player_controller_demo(10),
        enemymode="static",  # ai or static
        speed="fastest",  # normal or fastest
        randomini="yes",  # yes or no
    )

    f,p,e,t = environment.play(pcont=np.asarray(controller))
    return [f,p,e,t]

def run_best():
    enemies = [1,2,3,4,5,6,7,8]
    controller_path = 'assignment2/experiments/deap/controllers/'
    group = os.listdir(controller_path)
    test = False

    controllers_performance = []
    for controller in controllers:
        controller = np.loadtxt(controller_path + controller)
        enemy_scores = []
        for enemy in enemies:
            fitness = []
            for i in range(5):
                _, p, e, _ = run_single_enemy(enemy, controller)
                fitness.append(p-e)
            enemy_scores.append(np.mean(fitness))
        controllers_performance.append(np.mean(enemy_scores))
    
        #TODO: save performances to file or insert them to a boxplot function
    


            


if __name__ == '__main__':
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    run_best()
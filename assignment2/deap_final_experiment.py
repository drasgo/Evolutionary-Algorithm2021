import os
from deap import base
import numpy as np
import json
import pickle
import scipy

from plotting import line_plot, box_plot
from deap_main import main

from environment import New_Environment as Environment
from controllers.deap_controller import player_controller_demo

def evolve_agents(base_path, n=10):

    # Open the environment configuration
    config_file = 'assignment2/optimization_result.json'
    with open(config_file) as f:
        config = json.load(f)

    config['n_pop'] = 50 #50
    config['n_gens_ga'] = 50 #80
    config['n_gens_cma'] = 0 #20

    # Check folder for controllers
    controller_path = base_path + 'controllers/'
    if not os.path.exists(controller_path):
        os.makedirs(controller_path)

    # Check folder for images
    image_path = 'experiments/ga/images/'
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    # Check folder for results
    result_path = base_path + 'results/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Run the algorithm for both sets of agents
    agent_sets = [[6,8], [1,2,3,5]] 
    # agent_sets = [[6,8]]
    for i, enemies in enumerate(agent_sets):
        config['enemies'] = enemies

        # Check folder for controllers
        save_path = controller_path + 'group_' + str(i) + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        stats_per_run = []
        for j in range(n):
            print(f'Running: group: {enemies} run:{j}')
            experiment_name = 'run_' + str(j)
            config['name'] = experiment_name
            stats, best_ind = main(**config)
            stats_per_run.append(stats)
            np.savetxt(save_path + 'best_controller_' + experiment_name + '.txt', best_ind)

        with open(result_path + 'group_' + str(i) + '_evolution_result.pkl', "wb") as cp_file:
            pickle.dump(stats_per_run, cp_file)

def run_single_enemy(enemy, controller):
    '''
        A single controller fights a single enemy
    '''
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

def run_best(base_path):
    '''
        Test the controller performances, all 10 controllers fight each enemy 5 times. 
        It saves a 4D-matrix: 2 (n_groups) x 10 (controllers) x 8 enemies x 4 (average stats over 5 runs: fitness, player energy, enemy energy, time)
    '''
    enemies = [1,2,3,4,5,6,7,8]
    groups = ['group_0/', 'group_1/']
    controller_path = base_path + 'controllers/'
    results = np.empty([len(groups), 10, len(enemies), 4])
    for i, group in enumerate(groups):
        controllers = os.listdir(controller_path + group)
        for j, controller in enumerate(controllers):
            print(f'currently processing group {group} controller {j}')
            controller = np.loadtxt(controller_path + group + controller)
            for k, enemy in enumerate(enemies):
                results_single = np.empty((5, 4))
                for n in range(5):
                    f,p,e,t = run_single_enemy(enemy, controller)
                    results_single[n] = np.asarray([f,p,e,t])          
                results[i, j, k] = np.mean(results_single, axis=0)

    with open(base_path + 'results/performance_result.pkl', "wb") as cp_file:
        pickle.dump(results, cp_file)

def collect_datapoints(data):
    '''
        Extracts the 10 datapoints from the 4D-matrix for the boxplots.
    '''
    both_datapoints = []
    for group in range(2):
        datapoints = []
        for controller in range(10):
            gain = 0
            for enemy in range(8):
                gain += data[group, controller, enemy, 1] - data[group, controller, enemy, 2]
            datapoints.append(gain)
        both_datapoints.append(datapoints)
    return both_datapoints

def collect_champion(data_1, data_2):
    max_f = -np.inf
    best_stats = None
    best_description = None
    for i, file in enumerate([data_1, data_2]):
        for group in range(2):
            for controller in range(10):
                t = np.mean(file[group, controller, :, 0])
                if t > max_f:
                    max_f = t
                    best_stats = file[group, controller, :]
                    best_description = f'algorithm {i} group {group} controller {controller}'
    summary = np.mean(best_stats, axis=0)
    best_stats = np.vstack((best_stats, summary))                
    np.savetxt('assignment2/experiments/champion_table.csv', best_stats, delimiter=",", 
                header='Fitness, Player energy, Enemy energy, Time', fmt='%f')
    print('The overall best controller is: ', best_description)

def rename_lineplot(path):
    path_deap = 'assignment2/experiments/deap/'
    agent_sets = ['[6,8]', '[1,2,3,5]'] 
    for i, group in enumerate(['group_0', 'group_1']):
        file = path_deap + 'results/' + group + '_evolution_result.pkl'
        with open(file, 'rb') as f:
            data = pickle.load(f)
        line_plot('group ' +  str(agent_sets[i]), data, 'experiments/deap/images/')
    

def performance_to_boxplot(file_1, file_2):
    '''
        file_1: the file that belongs to the GA experiments
        file_2: the file that belongs to the GA-CMA-ES experiments
    '''
    # Read the performance file of GA
    with open(file_1, 'rb') as f:
        data_1 = pickle.load(f)
    # Read the performance file of GA-CMA-ES
    with open(file_2, 'rb') as f:
        data_2 = pickle.load(f)

    collect_champion(data_1, data_2)

    # Collect datapoints
    file_1_group_1_datapoints, file_1_group_2_datapoints = collect_datapoints(data_1)
    file_2_group_1_datapoints, file_2_group_2_datapoints = collect_datapoints(data_2)
    
    # Statistical significance
    print('Wilcoxon test group [6,8]: ', scipy.stats.wilcoxon(file_1_group_1_datapoints, file_2_group_1_datapoints))
    print('Wilcoxon test group [1,2,3,5]: ', scipy.stats.wilcoxon(file_1_group_2_datapoints, file_2_group_2_datapoints))

    # Create boxplots
    box_plot([6,8], [file_1_group_1_datapoints, file_2_group_1_datapoints], ("GA", "GA-CMS"), 'experiments')
    box_plot([1,2,3,5], [file_1_group_2_datapoints, file_2_group_2_datapoints], ("GA", "GA-CMS"), 'experiments')

if __name__ == '__main__':
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    path_ga = 'assignment2/experiments/ga/'
    path_deap = 'assignment2/experiments/deap/'
    # rename_lineplot(path_deap)
    # evolve_agents(path_ga)
    # run_best(path_ga)
    performance_to_boxplot(path_ga + 'results/performance_result.pkl', path_deap + 'results/performance_result.pkl')
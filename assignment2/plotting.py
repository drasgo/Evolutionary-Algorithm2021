import json
import os
import csv
import pickle
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, wilcoxon

font = {'family' : 'normal',
        'size'   : 12}
matplotlib.rc('font', **font)
local_dir = os.path.dirname(__file__)


def line_plot(experiment_name: str, total_fitnesses: List[List[List[float]]], images_folder: str="images"):
    """
    [           Gen1        Gen2            Gen3
       Run1: [[mean, max], [mean, max], [mean, max], ...],
       Run2: [[mean, max], [mean, max], ...],
       .
       .
       RunN: [[mean, max], ...]
    ]
    :param images_folder:
    :param experiment_name:
    :param total_fitnesses:
    :return:
    """
    if not os.path.exists(f"{local_dir}/{images_folder}/"):
        os.mkdir(f"{local_dir}/{images_folder}/")

    data = np.array(total_fitnesses)

    avg_mean = []
    avg_max = []
    # The number of generations is the second dimension, because this matrix will have shape
    # N_Runs x N_Gen x 2
    gens = list(range(1, data.shape[1] + 1))

    # Append in two lists one array for each generation, and each array has all the mean values(/max values)
    # for the considered generation
    for idx in range(data.shape[1]):
        avg_mean.append(data[:,idx, 0])
        avg_max.append(data[:,idx, 1])

    # Compute the average value of the mean/max for each generation
    average = np.mean(avg_mean, axis=1)
    maximum = np.max(avg_max, axis=1)
    # Compute the standard deviation for all the mean/max values for each generation
    std_average = np.std(avg_mean, axis=1)
    std_max = np.std(avg_max, axis=1)

    save_max = np.array(avg_max).tolist()
    save_avg = np.array(avg_mean).tolist()

    if not os.path.exists(f"{local_dir}/debug/"):
        os.mkdir(f"{local_dir}/debug/")

    save = {
        "max_values": save_max,
        "average_max_values": maximum.tolist(),
        "mean_values": save_avg,
        "average_mean_values": average.tolist()
    }

    with open(f"{local_dir}/debug/{experiment_name}.json", "w") as fp:
        json.dump(save, fp)
    # open(f"{local_dir}/{images_folder}/{experiment_name}_max_values.txt", "w").write(str(save_max))
    # open(f"{local_dir}/{images_folder}/{experiment_name}_average_max_values.txt", "w").write(str(maximum))
    #
    # open(f"{local_dir}/{images_folder}/{experiment_name}_mean_values.txt", "w").write(str(save_avg))
    # open(f"{local_dir}/{images_folder}/{experiment_name}_average_mean_values.txt", "w").write(str(average))

    plt.figure()
    plt.title(f"Enemy {experiment_name.replace('_', ' ')}")
    plt.xlabel("Generations")
    plt.ylabel("Fitness Value")
    plt.plot(gens, maximum, "k", color="red", label="Average of maximum fitness values")
    plt.plot(gens, average, "k", color="darkblue", label="Average of mean fitness values")
    plt.fill_between(gens, average - std_average, average + std_average, color="lightsteelblue")
    plt.fill_between(gens, maximum - std_max, maximum + std_max, color="lightsteelblue")
    plt.legend()
    #print(f"Line plotting results of {experiment_name}")
    plt.tight_layout(True)
    plt.savefig(f"{local_dir}/{images_folder}/{experiment_name}_line_plot.png", format="png")
    plt.clf()
    # average_fitnesses = np.mean(data, axis=1)
    # maximum_fitnesses = np.max(data, axis=1)
    # std_fit = np.std(data, axis=1)
    # generations = list(range(len(total_fitnesses)))

    # plt.plot(generations, maximum_fitnesses, "k", color="red", label="")
    # plt.plot(generations, average_fitnesses, "k", color="darkblue")
    # plt.fill_between(generations, average_fitnesses - std_fit, average_fitnesses + std_fit, color="lightsteelblue")


def box_plot(enemy: list, best_fitnesses, algorithm: Tuple=["GA", "GA-CMS"], images_folder: str="images"):
    """
    Each element is the mean value of 5 testing with best network from run N
    [ mean_value_run 1, mean_value_run 2, ..]
    :param experiment_name:
    :param best_fitnesses:
    :param images_folder:
    :return:
    """
    minimums = []
    for fitness in best_fitnesses:
        minimums.append(min(fitness))
    if not os.path.exists(f"{local_dir}/{images_folder}/"):
        os.mkdir(f"{local_dir}/{images_folder}/")
    plt.figure()
    axes = plt.axes()
    plt.title(f"Enemies {enemy}")
    plt.boxplot(best_fitnesses, positions = [1, 2], widths = 0.6)
    plt.ylabel("Gain")
    plt.xlabel("Algorithm")
    axes.set_xticklabels(algorithm)
    axes.set_xticks([1, 2])
    plt.tight_layout()
    plt.ylim(min(minimums)-1,100)
    #print(f"Box plotting results of Enemy {enemy}")
    plt.savefig(f"{local_dir}/{images_folder}/{enemy}_box_plot.png", format="png")
    plt.clf()


def plot_from_files(folder: str, enemies):
    files = os.listdir(f"{folder}/ga")
    box_results = []
    for enemy_list in enemies:
        enemy_string = f""
        for enemy in enemy_list:
            enemy_string += f"{enemy}_"
        enemy_files = [file for file in files if f"ga_solution_{enemy_string}" in file]
        lp_files = [file for file in enemy_files if "lpv" in file]
        bp_files = [file for file in enemy_files if "bpv" in file]

        # test_results = box_plot_from_files(folder, bp_files, enemy)
        # if test_results != 0:
        #    box_results.append([enemy, test_results])
        line_plot_from_files(folder, lp_files, enemy_string)
        make_controller_files(folder, bp_files, enemy_string)
    return box_results


def make_controller_files(folder: str, files, enemy_group):
    target_dir = f"{folder}/ga/controller"
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    for file in files:
        with open(f"{folder}/ga/{file}") as csvfile:
            reader = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                if row != []:
                    fitness = str(row[0]).replace(".", "x")
                    values = np.array(row[2].replace("[", "").replace("]", "").replace("\n", "").replace("  ", " ").replace("  ", " ").replace("  ", " ").strip().split(" "))
                    controller = []
                    for i in range(10):
                        controller.append(str(0.0))
                    for i in range(200):
                        controller.append(str(values[i]))
                    for i in range(5):
                        controller.append(str(0.0))
                    for i in range(50):
                        controller.append(str(values[i+200]))
                    print(controller)
        
        with open(f"{target_dir}/ga_solution_{enemy_group}{fitness}.txt", "w+") as txt_file:
            for value in controller:
                txt_file.write(f"{value}\n")
        txt_file.close()


# def box_plot_from_files(folder, files, enemy):
#     values = []
#     for file in files:
#         with open(f"{folder}/ga/{file}") as csvfile:
#             reader = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
#             for row in reader:
#                 if row != []:
#                     fitnesses = [float(value) for value in row[2][1:-2].split(", ")]
#                     values.append(sum(fitnesses) / len(fitnesses))

#     # with open(f"{folder}/deap/performance_result.pkl", 'rb') as deap_file:
#     #     deap_data = pickle.load(deap_file)
#     # print(deap_data)
#     neat_values = frame[enemy].tolist()[1:]
#     box_plot(enemy, [values, neat_values])
#     ttest = ttest_ind(values, neat_values).pvalue
#     mannwhitney = mannwhitneyu(values, neat_values).pvalue
#     #print(values, neat_values)
#     wilcoxonResult = wilcoxon(values, neat_values).pvalue
#     return [ttest, mannwhitney, wilcoxonResult]

def line_plot_from_files(folder, files, enemy_group):
    values = []
    idx = 0
    for file in files:
        values.append([])
        with open(f"{folder}/ga/{file}") as csvfile:
            reader = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                if row != []:
                    values[idx].append(row)
        idx += 1
    line_plot(f"{enemy_group}", values)


if __name__ == '__main__':
    # tot_test = [[1.,0.5,1.5], [2.,1.,3.], [3.,2.,4.]]
    # tot_test = [[[1., 0.5], [2.,5.], [3.,6.]],
    #             [[2., 1.5], [3.,4.], [2.5,5.5]]]
    # best_test = [0.4,2,2.5, 3.3]
    # line_plot("", tot_test)
    # box_plot("", best_test)
    enemies = [[6, 8], [1, 2, 3, 5]]
    print(plot_from_files(f"{os.path.dirname(__file__)}/results", enemies))


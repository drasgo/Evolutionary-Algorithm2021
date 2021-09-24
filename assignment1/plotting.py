import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np


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
    if not os.path.exists(images_folder + "/"):
        os.mkdir(images_folder)

    data = np.array(total_fitnesses)

    avg_mean = []
    avg_max = []
    # The number of generations is the second dimension, because this matrix will have shape
    # N_Runs x N_Gen x 2
    gens = list(range(data.shape[1]))

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

    plt.figure(0)
    plt.xlabel("Generations")
    plt.ylabel("Fitness Value")
    plt.plot(gens, maximum, "k", color="red", label="Average of maximum fitness values")
    plt.plot(gens, average, "k", color="darkblue", label="Average of mean fitness values")
    plt.fill_between(gens, average - std_average, average + std_average, color="lightsteelblue")
    plt.fill_between(gens, maximum - std_max, maximum + std_max, color="lightsteelblue")
    plt.legend()
    plt.savefig(f"{images_folder}/{experiment_name}_line_plot.png", format="png")

    # average_fitnesses = np.mean(data, axis=1)
    # maximum_fitnesses = np.max(data, axis=1)
    # std_fit = np.std(data, axis=1)
    # generations = list(range(len(total_fitnesses)))

    # plt.plot(generations, maximum_fitnesses, "k", color="red", label="")
    # plt.plot(generations, average_fitnesses, "k", color="darkblue")
    # plt.fill_between(generations, average_fitnesses - std_fit, average_fitnesses + std_fit, color="lightsteelblue")


def box_plot(experiment_name: str, best_fitnesses, images_folder: str="images"):
    """
    Each element is the mean value of 5 testing with best network from run N
    [ mean_value_run 1, mean_value_run 2, ..]
    :param experiment_name:
    :param best_fitnesses:
    :param images_folder:
    :return:
    """
    if not os.path.exists(images_folder + "/"):
        os.mkdir(images_folder)
    data = np.array(best_fitnesses)
    plt.figure(1)
    plt.boxplot(data)
    plt.ylabel("Individual Gain")
    plt.xlabel(experiment_name)
    plt.savefig(f"{images_folder}/{experiment_name}_box_plot.png", format="png")


if __name__ == '__main__':
    # tot_test = [[1.,0.5,1.5], [2.,1.,3.], [3.,2.,4.]]
    tot_test = [[[1., 0.5], [2.,5.], [3.,6.]],
                [[2., 1.5], [3.,4.], [2.5,5.5]]]
    best_test = [0.4,2,2.5, 3.3]
    line_plot("", tot_test)
    box_plot("", best_test)



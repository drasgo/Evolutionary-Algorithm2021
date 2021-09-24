import os

import matplotlib.pyplot as plt
import numpy as np


def plot(experiment_name: str, total_fitnesses: list, images_folder: str="img"):
    """
    [
        Gen1[fitness_value_for_agent1,
        fitness_value_for_agent2, ..],

        Gen2[fitness_value_for_agent1,
        fitness_value_for_agent2, ..]
    ]
    :param images_folder:
    :param experiment_name:
    :param total_fitnesses:
    :return:
    """
    if not os.path.exists(images_folder + "/"):
        os.mkdir(images_folder)

    data = np.array(total_fitnesses)
    average_fitnesses = np.mean(data, axis=1)
    generations = list(range(len(total_fitnesses)))
    std_fit = np.std(data, axis=1)

    plt.figure(0)
    plt.plot(generations, average_fitnesses, "k", color="darkblue")
    plt.fill_between(generations, average_fitnesses - std_fit, average_fitnesses + std_fit, color="lightsteelblue")
    plt.savefig(f"{images_folder}/{experiment_name}_line_plot.png", format="png")

    plt.figure(1)
    plt.boxplot(data)
    plt.savefig(f"{images_folder}/{experiment_name}_box_plot.png", format="png")


if __name__ == '__main__':
    tot_test = [[1.,0.5,1.5], [2.,1.,3.], [3.,2.,4.]]
    plot("", tot_test)



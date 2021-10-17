import pickle
from plotting import line_plot, box_plot

def check_results():
    result_path = 'assignment2/experiments/deap/results/'
    image_path = 'assignment2/experiments/deap/results/test/'
    with open(result_path + 'performance_result.pkl', "rb") as cp_file:
        results = pickle.load(cp_file)

    print(results)

def f_score(a,b,c,case):
    return (100 - case[1]) ** a - (100 - case[0]) ** b - case[2] ** c


def f():
    a = 2
    b = 1.5
    c = 1.15
    best = [100, 0, 30] # p, e, avg_life, time
    worst = [0, 100, 3000]
    max = f_score(a,b,c, best)
    min = f_score(a,b,c,worst)
    print(max, min)

def rename_plots():
    file = 'assignment2/experiments/deap/results/group_0_evolution_result.pkl'
    with open(file, 'rb') as f:
        data = pickle.load(f)
    line_plot('group [6,8]', data, 'experiments/deap/images/')

    file = 'assignment2/experiments/deap/results/group_1_evolution_result.pkl'
    with open(file, 'rb') as f:
        data = pickle.load(f)
    line_plot('group [1,2,3,5]', data, 'experiments/deap/images/')

if __name__ == '__main__':
    rename_plots()

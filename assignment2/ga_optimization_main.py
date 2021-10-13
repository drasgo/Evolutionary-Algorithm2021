import os

from numpy.ma import MaskedArray
import sklearn.utils.fixes
sklearn.utils.fixes.MaskedArray = MaskedArray

from skopt import gp_minimize
from assignment2.controllers.gann_controller import ga_controller as Controller

def optimize(runs: int = 5):
    for run in range(runs):
        print(f"Run {run}: {gp_minimize(single_run, [(0.0, 0.5), (0.0001, 1.0), (0.0, 1.0)])}")

def single_run(inputs) -> float:
    return Controller(50, 20, [5, 8], inputs[0], inputs[1], inputs[2]).execute()

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

if __name__ == '__main__':
    optimize()
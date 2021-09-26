import os

from typing import List

from assignment1.controllers.gann_controller import ga_controller as Controller
from assignment1.environment import New_Environment as Environment

def run(population: int, generations: int, enemies: List[int]):
    Controller(population, generations, enemies).execute()

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

run(150, 50, [1])
run(150, 50, [2])
run(150, 50, [3])
run(150, 50, [4])
run(150, 50, [5])
run(150, 50, [6])
run(150, 50, [7])
run(150, 50, [8])


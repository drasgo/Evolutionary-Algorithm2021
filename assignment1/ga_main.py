import os

from typing import List

from assignment1.controllers.gann_controller import ga_controller as Controller
from assignment1.environment import New_Environment as Environment

def run(population: int, generations: int, enemies: List[int]):
    Controller(population, generations, enemies).algorithm.run()

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

run(150, 50, [1])


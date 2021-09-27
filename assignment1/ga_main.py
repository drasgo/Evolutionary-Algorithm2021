import os

from typing import List

from assignment1.controllers.gann_controller import ga_controller as Controller
from assignment1.environment import New_Environment as Environment

def run(population: int, generations: int, enemies: List[int]):
    return Controller(population, generations, enemies).execute()

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

for enemy in [2, 4, 5]:
    best_fitness = run(5, 5, [enemy])[1]
    print(f"Best fitness for enemy {enemy} was {best_fitness}")


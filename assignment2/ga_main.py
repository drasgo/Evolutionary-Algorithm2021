import os
from typing import List

from assignment2.controllers.gann_controller import ga_controller as Controller

def run(cycles: int, enemies: List[List[int]]):
    for enemy_list in enemies:
        for idx in range(cycles):
            Controller(150, 20, enemy_list, idx, 0.01, 0.2, 0.8).execute()

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

if __name__ == '__main__':
    run(10, [[6, 8], [1, 2, 3, 5]])

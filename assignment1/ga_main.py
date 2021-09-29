import os
from typing import List

from assignment1.controllers.gann_controller import ga_controller as Controller

def run(cycles: int, enemies: List[int]):
    for enemy in enemies:
        for idx in range(cycles):
            Controller(150, 20, [enemy]).execute()

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

if __name__ == '__main__':
    run(10, [2, 5, 8])

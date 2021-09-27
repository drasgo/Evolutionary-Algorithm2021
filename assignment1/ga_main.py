import os
from typing import List

from assignment1.controllers.gann_controller import ga_controller as Controller

def run():
    for enemy in {2, 4, 5}:
        Controller(5, 5, [enemy]).execute()

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

if __name__ == '__main__':
    run()

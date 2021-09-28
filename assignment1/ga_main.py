import os

from assignment1.controllers.gann_controller import ga_controller as Controller

def run(cycles: int):
    for idx in range(cycles):
        for enemy in {2, 4, 5}:
            Controller(150, 20, [enemy]).execute()

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

if __name__ == '__main__':
    run(10)

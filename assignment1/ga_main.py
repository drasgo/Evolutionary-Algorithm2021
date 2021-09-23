import os

from assignment1.controllers.ga_controller import ga_controller as controller
from assignment1.environment import Environment

n_hidden_neurons = 20

experiment_name = "ga_specialist"

env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

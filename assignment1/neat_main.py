import os

from assignment1.neat_controller import player_controller
from evoman.environment import Environment


def evolve(model, fitness_values):
    pass


if __name__ == '__main__':
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    n_hidden_neurons = 20
    experiment_name = "neat_specialist"
    env = Environment(experiment_name=experiment_name,
                      enemies=[2],
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest")


import os

from assignment1.neat_controller import Neat_Controller
from assignment1.environment import Environment


if __name__ == '__main__':
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "configs", "config-neat")

    experiment_name = "neat_specialist"
    env = Environment(experiment_name=experiment_name,
                      enemies=[2],
                      playermode="ai",
                      player_controller=Neat_Controller(config_path),
                      enemymode="static",
                      level=2,
                      speed="fastest")


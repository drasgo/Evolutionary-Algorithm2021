from ga_controller import player_controller
from evoman.environment import Environment

n_hidden_neurons = 20

experiment_name = "ga_specialist"

env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

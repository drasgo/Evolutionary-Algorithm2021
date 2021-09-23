# implements controller structure for player
from evoman.controller import Controller
import numpy as np
import neat


class Neat_Controller(Controller):
	def __init__(self, neat_configs: str):
		# Load config file
		config = self.load_config(neat_configs)
		# Create the population, which is the top-level object for a NEAT run.
		self.population = self.prepare_population(configs=config)
		self.current_winner_net = self.population.best_genome
		self.inputs = []
		self.outputs = []
		self.fitnesses = []

	def load_config(self, config_path: str):
		try:
			return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
								 neat.DefaultSpeciesSet, neat.DefaultStagnation,
								 config_path)
		except Exception as exc:
			print(f"Failed loading neat config file {config_path}, with exception {exc}")
			quit()

	def prepare_population(self, configs) -> neat.Population:
		population = neat.Population(configs)
		# Add a stdout reporter to show progress in the terminal.
		population.add_reporter(neat.StdOutReporter(True))
		stats = neat.StatisticsReporter()
		population.add_reporter(stats)
		population.add_reporter(neat.Checkpointer(5))
		return population

	def control(self, inputs: np.ndarray, controller=None):
		"""
		Here what action is going to do is decided. This is called by the function update in the class Player
		which is called automatically when updating the tilemaps at each tick of the game by the function run_single
		of the class Environment.
		:param inputs: A numpy list containing the values of the sensors available.
		Default array contains (in this order):
		 - horizontal distance between player and enemy
		 - vertical distance between player and enemy
		 - direction of player (1 -> right, -1 -> left)
		 - direction of enemy (1 -> right, -1 -> left)
		 - FOR EACH OF THE BULLETS (default there are 8 bullets total)
		 	- horizontal distance between player and bullet
		 	- vertical distance between player and bullet
		 	(NOTE: As a default, if there were fired less than 8 bullets, the remaining bullets will be added
		 	here with horizontal and vertical distance of 0)
		:param controller: What is going to be used here as controller. It can be whatever, and outside of the environment
		this will be used for updating its inner values (e.g. with evolutionary algorithms), and here it is used for
		extracting the next move. E.g. it can be a dictionary containing the weights and biases of a neural network,
		or a pytorch object, and so on.
		:return: a list of 5 values as one-hot vector (all 0s except one 1):
			- left,
			- right,
			- jump,
			- shoot,
			- release.
		"""
		output = self.current_winner_net.activate(inputs)
		return [1 if com > 0.5 else 0 for com in output]

	def eval_genomes(self, genomes):
		for genome_id, genome in genomes:
			genome.fitness = 4.0
			# net = neat.nn.FeedForwardNetwork.create(genome, config)
			for fit in self.fitnesses:
				genome.fitness -= fit
			# for inps, outs in zip(self.inputs, self.outputs):

	def evolve(self, fitness_value: float):
		self.fitnesses.append(fitness_value)
		self.current_winner_net = self.population.run(self.eval_genomes, 300)

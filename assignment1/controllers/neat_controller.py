# implements controller structure for player
from evoman.controller import Controller
import numpy as np
import neat


class Neat_Controller(Controller):
	def __init__(self, neat_configs: str):
		# Load config file
		self.configs = self.load_config(neat_configs)
		# Create the population, which is the top-level object for a NEAT run.
		self.stats = neat.StatisticsReporter()
		self.population = self.prepare_population()
		self.current_winner_net = self.population.best_genome
		self.nets = []

	def prepare_new_networks(self):
		self.nets = []
		for gen_idx, (_, genome) in enumerate(list(self.population.population.items())):
			self.nets.append([gen_idx, genome, neat.nn.FeedForwardNetwork.create(genome, self.configs)])

	def load_config(self, config_path: str):
		try:
			return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
							   neat.DefaultSpeciesSet, neat.DefaultStagnation,
							   config_path)
		except Exception as exc:
			print(f"Failed loading neat config file {config_path}, with exception {exc}")
			quit()

	def prepare_population(self) -> neat.Population:
		population = neat.Population(self.configs)
		# Add a stdout reporter to show progress in the terminal.
		population.add_reporter(neat.StdOutReporter(True))
		population.add_reporter(self.stats)
		population.add_reporter(neat.Checkpointer(10))
		return population

	def control(self, inputs: np.ndarray, controller=None, fitness: float=None):
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
		In this instance, this whole class is the controller, so we don't really need it
		:param fitness: current fitness score, related to the current sensors and the current historical episode situation.
		:return: a list of 5 values as one-hot vector (all 0s except one 1):
			- left,
			- right,
			- jump,
			- shoot,
			- release.
		"""
		network = controller[2]
		output = network.activate(inputs)
		action = [1 if act > 0.5 else 0 for act in output]

		# OPPURE SI SALVA L'AZIONE, E POI DOPO SI TESTA L'AZIONE CON L'OUTPUT, E SI CALCOLA UN ALTRO FITNESS (PER OGNI AZIONE)
		return action

	def eval_genomes(self, genomes, config):
		for idx, net in enumerate(self.nets):
			print(f"Evolving network n°. {idx}")
			gen_idx = net[0]
			fit = net[-1]
			genomes[gen_idx][1].fitness = fit
		# 	OTHER POSSIBLE FITNESS FUNCTION IS: SUM OF(value of index of output - fitness)^2


	def evolve(self):
		print("Starting evolving")
		_ = self.population.run(self.eval_genomes, 1)

	def reset(self):
		self.inputs = []
		self.actions = []
		self.fitnesses = []